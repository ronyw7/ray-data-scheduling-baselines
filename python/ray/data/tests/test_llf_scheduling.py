"""Tests for LLF (Least-Laxity-First) scheduling policy.

Covers:
- Version 1 (t_M=0, L=0): collapses to stage-by-stage ordering
- Version 2 (t_M=i*T):    deadline-based scheduling with simulated arrivals
- EDF variant:             deadline without self-cost C_oM
- Integration tests with real Ray pipelines
- Synthetic benchmark
"""

import time
from unittest.mock import MagicMock

import pytest

import ray
from ray.data._internal.execution.interfaces import (
    ExecutionOptions,
    ExecutionResources,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    create_map_transformer_from_block_fn,
)
from ray.data._internal.execution.streaming_executor_state import (
    build_streaming_topology,
    select_operator_to_run,
    _compute_c_path,
    _compute_min_deadline,
    _get_t_m,
    _get_latency_target,
)
from ray.data._internal.execution.util import make_ref_bundles
from ray.data.context import DataContext


def make_map_transformer(block_fn):
    def map_fn(block_iter):
        for block in block_iter:
            yield block_fn(block)

    return create_map_transformer_from_block_fn(map_fn)


def make_ref_bundle(x):
    return make_ref_bundles([[x]])[0]


def mock_resource_manager(global_limits=None, global_usage=None):
    empty_resource = ExecutionResources(0, 0, 0)
    global_limits = global_limits or empty_resource
    global_usage = global_usage or empty_resource
    return MagicMock(
        get_global_limits=MagicMock(return_value=global_limits),
        get_global_usage=MagicMock(return_value=global_usage),
        get_downstream_fraction=MagicMock(return_value=0.0),
        get_downstream_object_store_memory=MagicMock(return_value=0),
        op_resource_allocator_enabled=MagicMock(return_value=False),
    )


def mock_autoscaler():
    return MagicMock()


def _set_profiled_duration(op, avg_duration, num_tasks=5):
    """Helper to mock profiled task duration for an operator."""
    op._metrics.block_generation_time = avg_duration * num_tasks
    op._metrics.num_tasks_finished = num_tasks


# ---------------------------------------------------------------------------
# Test _get_t_m
# ---------------------------------------------------------------------------


class TestGetTm:
    def test_v1_always_zero(self):
        bundle = make_ref_bundle("x")
        bundle.partition_index = 42
        assert _get_t_m(bundle, "llf_v1", inter_arrival_time=0.1) == 0.0

    def test_v2_uses_index(self):
        bundle = make_ref_bundle("x")
        bundle.partition_index = 5
        assert _get_t_m(bundle, "llf_v2", inter_arrival_time=0.2) == 1.0  # 5 * 0.2

    def test_edf_uses_index(self):
        bundle = make_ref_bundle("x")
        bundle.partition_index = 10
        assert _get_t_m(bundle, "edf", inter_arrival_time=0.5) == 5.0  # 10 * 0.5

    def test_v2_none_index_defaults_to_zero(self):
        bundle = make_ref_bundle("x")
        bundle.partition_index = None
        assert _get_t_m(bundle, "llf_v2", inter_arrival_time=0.1) == 0.0


# ---------------------------------------------------------------------------
# Test _get_latency_target
# ---------------------------------------------------------------------------


class TestGetLatencyTarget:
    def test_v1_always_zero(self):
        inputs = make_ref_bundles([[x] for x in range(3)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        topo, _ = build_streaming_topology(o2, ExecutionOptions())
        assert _get_latency_target("llf_v1", topo, None) == 0.0

    def test_v2_explicit_target(self):
        inputs = make_ref_bundles([[x] for x in range(3)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        topo, _ = build_streaming_topology(o2, ExecutionOptions())
        assert _get_latency_target("llf_v2", topo, 5.0) == 5.0

    def test_v2_auto_computed(self):
        inputs = make_ref_bundles([[x] for x in range(3)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        topo, _ = build_streaming_topology(o2, ExecutionOptions())
        _set_profiled_duration(o2, 2.0)
        L = _get_latency_target("llf_v2", topo, None)
        # L = 0.0 + 2.0 = 2.0
        assert L == 2.0


# ---------------------------------------------------------------------------
# Test _compute_c_path
# ---------------------------------------------------------------------------


class TestComputeCPath:
    def test_linear_pipeline(self):
        inputs = make_ref_bundles([[x] for x in range(5)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        o3 = MapOperator.create(
            make_map_transformer(lambda block: block), o2
        )
        topo, _ = build_streaming_topology(o3, ExecutionOptions())

        _set_profiled_duration(o2, 1.0)
        _set_profiled_duration(o3, 2.0)

        c_path = _compute_c_path(topo)

        assert c_path[o3] == 0.0
        assert c_path[o2] == 2.0   # C_oM[o3] + C_path[o3]
        assert c_path[o1] == 3.0   # C_oM[o2] + C_path[o2]

    def test_no_profiling_data(self):
        inputs = make_ref_bundles([[x] for x in range(5)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        topo, _ = build_streaming_topology(o2, ExecutionOptions())

        c_path = _compute_c_path(topo)
        assert c_path[o1] == 0.0
        assert c_path[o2] == 0.0


# ---------------------------------------------------------------------------
# Test Version 1: t_M=0, L=0 collapses to stage-by-stage
# ---------------------------------------------------------------------------


class TestLLFv1:
    def test_v1_selects_upstream_first(self):
        """With t_M=0, L=0: ddl_M = -C_oM - C_path.
        The operator with the largest (C_oM + C_path) has the most negative
        deadline (smallest laxity), so it gets selected first.
        For a linear pipeline, that's the operator furthest from sink.
        """
        inputs = make_ref_bundles([[x] for x in range(20)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        o3 = MapOperator.create(
            make_map_transformer(lambda block: block), o2
        )
        topo, _ = build_streaming_topology(o3, ExecutionOptions())
        resource_manager = mock_resource_manager(
            global_limits=ExecutionResources.for_limits(1, 1, 1),
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources(0, 0, 0)
        )

        # Give all ops profiled durations of 1.0s.
        _set_profiled_duration(o2, 1.0)
        _set_profiled_duration(o3, 1.0)

        # Both o2 and o3 have bundles queued.
        b1 = make_ref_bundle("b1")
        b1.partition_index = 0
        topo[o1].outqueue.append(b1)

        b2 = make_ref_bundle("b2")
        b2.partition_index = 0
        topo[o2].outqueue.append(b2)

        # With v1: ddl for o2 = 0 + 0 - 1.0 - 1.0 = -2.0
        #          ddl for o3 = 0 + 0 - 1.0 - 0.0 = -1.0
        # Laxity for o2 < laxity for o3, so o2 selected (upstream first).
        selected = select_operator_to_run(
            topo, resource_manager, [], mock_autoscaler(), True,
            scheduling_policy="llf_v1",
        )
        assert selected == o2

    def test_v1_is_deterministic_regardless_of_partition_index(self):
        """V1 ignores partition_index — all partitions have same t_M=0."""
        inputs = make_ref_bundles([[x] for x in range(20)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        o3 = MapOperator.create(
            make_map_transformer(lambda block: block), o2
        )
        topo, _ = build_streaming_topology(o3, ExecutionOptions())
        resource_manager = mock_resource_manager(
            global_limits=ExecutionResources.for_limits(1, 1, 1),
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources(0, 0, 0)
        )
        _set_profiled_duration(o2, 1.0)
        _set_profiled_duration(o3, 1.0)

        # o2 has partition 999, o3 has partition 0 — shouldn't matter for v1.
        b1 = make_ref_bundle("b1")
        b1.partition_index = 999
        topo[o1].outqueue.append(b1)

        b2 = make_ref_bundle("b2")
        b2.partition_index = 0
        topo[o2].outqueue.append(b2)

        selected = select_operator_to_run(
            topo, resource_manager, [], mock_autoscaler(), True,
            scheduling_policy="llf_v1",
        )
        # Same result as above — v1 is insensitive to partition index.
        assert selected == o2


# ---------------------------------------------------------------------------
# Test Version 2: t_M = i * T
# ---------------------------------------------------------------------------


class TestLLFv2:
    def test_v2_older_partition_more_urgent(self):
        """Partition with smaller index has earlier simulated arrival,
        so tighter deadline, so lower laxity."""
        inputs = make_ref_bundles([[x] for x in range(20)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        o3 = MapOperator.create(
            make_map_transformer(lambda block: block), o2
        )
        topo, _ = build_streaming_topology(o3, ExecutionOptions())
        resource_manager = mock_resource_manager(
            global_limits=ExecutionResources.for_limits(1, 1, 1),
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources(0, 0, 0)
        )

        _set_profiled_duration(o2, 1.0)
        _set_profiled_duration(o3, 1.0)

        # o2 has an OLD partition (index=1), o3 has a NEW partition (index=100).
        old = make_ref_bundle("old")
        old.partition_index = 1
        topo[o1].outqueue.append(old)

        new = make_ref_bundle("new")
        new.partition_index = 100
        topo[o2].outqueue.append(new)

        # t_M for o2's bundle = 1 * 0.1 = 0.1
        # t_M for o3's bundle = 100 * 0.1 = 10.0
        # ddl for o2 = 0.1 + L - 1.0 - 1.0 = 0.1 + L - 2.0
        # ddl for o3 = 10.0 + L - 1.0 - 0.0 = 10.0 + L - 1.0
        # o2's deadline is tighter → selected.
        selected = select_operator_to_run(
            topo, resource_manager, [], mock_autoscaler(), True,
            scheduling_policy="llf_v2",
            llf_inter_arrival_time=0.1,
            llf_latency_target=5.0,
        )
        assert selected == o2

    def test_v2_downstream_preferred_when_same_age(self):
        """For same-age partitions, downstream op has smaller C_path,
        so a LATER deadline, so HIGHER laxity. But with same t_M and
        C_oM, the upstream op has tighter deadline due to larger C_path."""
        inputs = make_ref_bundles([[x] for x in range(20)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        o3 = MapOperator.create(
            make_map_transformer(lambda block: block), o2
        )
        topo, _ = build_streaming_topology(o3, ExecutionOptions())
        resource_manager = mock_resource_manager(
            global_limits=ExecutionResources.for_limits(1, 1, 1),
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources(0, 0, 0)
        )

        _set_profiled_duration(o2, 1.0)
        _set_profiled_duration(o3, 1.0)

        # Same partition index for both.
        b1 = make_ref_bundle("b1")
        b1.partition_index = 5
        topo[o1].outqueue.append(b1)

        b2 = make_ref_bundle("b2")
        b2.partition_index = 5
        topo[o2].outqueue.append(b2)

        # ddl for o2 = 0.5 + L - 1.0 - 1.0 = 0.5 + L - 2.0
        # ddl for o3 = 0.5 + L - 1.0 - 0.0 = 0.5 + L - 1.0
        # o2 has tighter deadline → upstream first (same as v1 for equal ages).
        selected = select_operator_to_run(
            topo, resource_manager, [], mock_autoscaler(), True,
            scheduling_policy="llf_v2",
            llf_inter_arrival_time=0.1,
            llf_latency_target=5.0,
        )
        assert selected == o2


# ---------------------------------------------------------------------------
# Test EDF
# ---------------------------------------------------------------------------


class TestEDF:
    def test_edf_ignores_self_cost(self):
        """EDF omits C_oM: ddl = t_M + L - C_path."""
        inputs = make_ref_bundles([[x] for x in range(20)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        o3 = MapOperator.create(
            make_map_transformer(lambda block: block), o2
        )
        topo, _ = build_streaming_topology(o3, ExecutionOptions())
        resource_manager = mock_resource_manager(
            global_limits=ExecutionResources.for_limits(1, 1, 1),
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources(0, 0, 0)
        )

        # o2 has very high C_oM=10s, o3 has low C_oM=0.1s.
        _set_profiled_duration(o2, 10.0)
        _set_profiled_duration(o3, 0.1)

        # Same partition index, same t_M.
        b1 = make_ref_bundle("b1")
        b1.partition_index = 1
        topo[o1].outqueue.append(b1)

        b2 = make_ref_bundle("b2")
        b2.partition_index = 1
        topo[o2].outqueue.append(b2)

        # EDF: ddl for o2 = t_M + L - C_path[o2] = 0.1 + L - 0.1
        #      ddl for o3 = t_M + L - C_path[o3] = 0.1 + L - 0.0
        # C_path[o2] = C_oM[o3] = 0.1, C_path[o3] = 0
        # o2 has tighter deadline → selected.
        selected = select_operator_to_run(
            topo, resource_manager, [], mock_autoscaler(), True,
            scheduling_policy="edf",
            llf_inter_arrival_time=0.1,
            llf_latency_target=5.0,
        )
        assert selected == o2


# ---------------------------------------------------------------------------
# Test metadata ops and empty topology
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_metadata_ops_still_prioritized(self):
        inputs = make_ref_bundles([[x] for x in range(20)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        o3 = MapOperator.create(
            make_map_transformer(lambda block: block), o2
        )
        topo, _ = build_streaming_topology(o3, ExecutionOptions())
        resource_manager = mock_resource_manager(
            global_limits=ExecutionResources.for_limits(1, 1, 1),
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources(0, 0, 0)
        )
        _set_profiled_duration(o2, 1.0)
        _set_profiled_duration(o3, 1.0)

        b1 = make_ref_bundle("b1")
        b1.partition_index = 0
        topo[o1].outqueue.append(b1)

        b2 = make_ref_bundle("b2")
        b2.partition_index = 0
        topo[o2].outqueue.append(b2)

        # o3 is metadata-only → should be selected despite less urgent laxity.
        o3.throttling_disabled = MagicMock(return_value=True)

        selected = select_operator_to_run(
            topo, resource_manager, [], mock_autoscaler(), True,
            scheduling_policy="llf_v2",
            llf_inter_arrival_time=0.1,
            llf_latency_target=5.0,
        )
        assert selected == o3

    def test_empty_topology(self):
        inputs = make_ref_bundles([[x] for x in range(5)])
        o1 = InputDataBuffer(inputs)
        o2 = MapOperator.create(
            make_map_transformer(lambda block: block), o1
        )
        topo, _ = build_streaming_topology(o2, ExecutionOptions())
        resource_manager = mock_resource_manager(
            global_limits=ExecutionResources.for_limits(1, 1, 1),
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources(0, 0, 0)
        )

        selected = select_operator_to_run(
            topo, resource_manager, [], mock_autoscaler(), True,
            scheduling_policy="llf_v1",
        )
        assert selected is None


# ---------------------------------------------------------------------------
# Integration tests with real Ray pipelines
# ---------------------------------------------------------------------------


def test_llf_v1_integration():
    """V1 should produce correct output (all data processed)."""
    ctx = DataContext.get_current()
    original = ctx.scheduling_policy
    try:
        ctx.scheduling_policy = "llf_v1"
        ds = ray.data.range(100).map(lambda row: {"id": row["id"] * 2})
        results = sorted([row["id"] for row in ds.take_all()])
        assert results == sorted([x * 2 for x in range(100)])
    finally:
        ctx.scheduling_policy = original


def test_llf_v2_integration():
    """V2 should produce correct output."""
    ctx = DataContext.get_current()
    original_policy = ctx.scheduling_policy
    original_T = ctx.llf_inter_arrival_time
    original_L = ctx.llf_latency_target
    try:
        ctx.scheduling_policy = "llf_v2"
        ctx.llf_inter_arrival_time = 0.01
        ctx.llf_latency_target = 10.0

        ds = ray.data.range(100)
        ds = ds.map(lambda row: {"id": row["id"] * 2})
        ds = ds.map(lambda row: {"id": row["id"] + 1})
        results = sorted([row["id"] for row in ds.take_all()])
        assert results == sorted([x * 2 + 1 for x in range(100)])
    finally:
        ctx.scheduling_policy = original_policy
        ctx.llf_inter_arrival_time = original_T
        ctx.llf_latency_target = original_L


def test_edf_integration():
    """EDF should produce correct output."""
    ctx = DataContext.get_current()
    original = ctx.scheduling_policy
    try:
        ctx.scheduling_policy = "edf"
        ctx.llf_latency_target = 10.0
        ds = ray.data.range(50).map(lambda row: {"id": row["id"] * 3})
        results = sorted([row["id"] for row in ds.take_all()])
        assert results == sorted([x * 3 for x in range(50)])
    finally:
        ctx.scheduling_policy = original
        ctx.llf_latency_target = None


def test_default_unchanged():
    """Default policy still works."""
    ctx = DataContext.get_current()
    original = ctx.scheduling_policy
    try:
        ctx.scheduling_policy = None
        ds = ray.data.range(50).map(lambda row: {"id": row["id"] + 10})
        results = sorted([row["id"] for row in ds.take_all()])
        assert results == sorted([x + 10 for x in range(50)])
    finally:
        ctx.scheduling_policy = original


# ---------------------------------------------------------------------------
# Synthetic benchmark
# ---------------------------------------------------------------------------


def test_synthetic_benchmark():
    """3-stage pipeline: Load -> Transform -> Inference.
    Runs with default, llf_v1, and llf_v2 policies.
    """
    import numpy as np

    ctx = DataContext.get_current()
    original_policy = ctx.scheduling_policy
    original_T = ctx.llf_inter_arrival_time
    original_L = ctx.llf_latency_target

    results = {}
    for policy in [None, "llf_v1", "llf_v2"]:
        try:
            ctx.scheduling_policy = policy
            if policy == "llf_v2":
                ctx.llf_inter_arrival_time = 0.01
                ctx.llf_latency_target = 5.0

            ds = ray.data.range(40)

            def load_fn(row):
                row["data"] = np.zeros(1000)
                return row

            def transform_fn(row):
                time.sleep(0.01)
                row["data"] = row["data"] + 1
                return row

            def inference_fn(row):
                time.sleep(0.02)
                row["result"] = float(row["data"].sum())
                return row

            ds = ds.map(load_fn).map(transform_fn).map(inference_fn)

            start = time.time()
            output = ds.take_all()
            elapsed = time.time() - start

            assert len(output) == 40
            for row in output:
                assert row["result"] == 1000.0

            policy_name = policy or "default"
            results[policy_name] = elapsed
            print(f"Policy={policy_name}: {elapsed:.2f}s")
        finally:
            ctx.scheduling_policy = original_policy
            ctx.llf_inter_arrival_time = original_T
            ctx.llf_latency_target = original_L

    assert len(results) == 3
    for name, t in results.items():
        print(f"  {name}: {t:.2f}s")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
