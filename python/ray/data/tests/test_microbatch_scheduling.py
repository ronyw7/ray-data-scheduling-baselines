"""Unit tests for the microbatch (BSP) scheduling policy.

Covers:
  - InputDataBuffer stamps partition_index on bundles.
  - The microbatch helpers `_compute_depth_map`, `_global_min_epoch`,
    `_op_has_pending_at_epoch`, `_transitive_ancestors`.
  - `_microbatch_select` enforces intra-batch stage and inter-batch barriers.
  - Drizzle toggles: group_size (inter-batch) and stage_barrier (intra-batch).
"""

import pytest

import ray
from ray.data._internal.execution.interfaces import (
    ExecutionOptions,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    create_map_transformer_from_block_fn,
)
from ray.data._internal.execution.streaming_executor_state import (
    _compute_depth_map,
    _global_min_epoch,
    _microbatch_select,
    _op_has_pending_at_epoch,
    _transitive_ancestors,
    build_streaming_topology,
)
from ray.data._internal.execution.util import make_ref_bundles
from ray.data.tests.conftest import *  # noqa


def _make_map_transformer(block_fn):
    def map_fn(block_iter):
        for block in block_iter:
            yield block_fn(block)

    return create_map_transformer_from_block_fn(map_fn)


def _linear_topology(num_ops=3):
    """Build a linear DAG: InputDataBuffer -> map1 -> map2 -> ...."""
    inputs = make_ref_bundles([[x] for x in range(8)])
    src = InputDataBuffer(inputs)
    tail = src
    ops = [src]
    for _ in range(num_ops):
        tail = MapOperator.create(
            _make_map_transformer(lambda block: block), tail
        )
        ops.append(tail)
    topo, _ = build_streaming_topology(tail, ExecutionOptions())
    return topo, ops


class _FakeTask:
    """Minimal stand-in for DataOpTask that the barrier helpers can read.

    The in-flight barrier check iterates `op.get_active_tasks()` and reads
    `task.partition_index`, so that's all we need to fake.
    """

    def __init__(self, partition_index: int, task_index: int):
        self.partition_index = partition_index
        self.task_index = task_index


def _inject_inflight(op: MapOperator, partition_indices):
    """Test helper: attach fake in-flight tasks to a MapOperator.

    Mirrors what `MapOperator._submit_data_task` would do when stamping
    DataOpTask.partition_index from the input bundle. Avoids building real
    streaming generators.
    """
    start = max(op._data_tasks.keys(), default=-1) + 1
    for k, pid in enumerate(partition_indices):
        op._data_tasks[start + k] = _FakeTask(pid, start + k)


def test_input_data_buffer_stamps_partition_index():
    inputs = make_ref_bundles([[x] for x in range(4)])
    for bundle in inputs:
        assert bundle.partition_index is None
    src = InputDataBuffer(inputs)
    src.start(ExecutionOptions())
    seen = []
    while src.has_next():
        bundle = src._get_next_inner()
        seen.append(bundle.partition_index)
    assert seen == [0, 1, 2, 3]


def test_compute_depth_map_linear():
    topo, ops = _linear_topology(num_ops=3)
    depth = _compute_depth_map(topo)
    for i, op in enumerate(ops):
        assert depth[op] == i


def test_global_min_epoch_uses_inqueue_and_inflight():
    topo, ops = _linear_topology(num_ops=2)
    _, m1, m2 = ops
    B = 2
    # Put bundles with partition_index 2,3 in m1's inqueue (epoch 1)
    # and partition_index 0 in m2's in-flight (epoch 0).
    r1 = make_ref_bundles([["a"]])[0]
    r1.partition_index = 2
    r2 = make_ref_bundles([["b"]])[0]
    r2.partition_index = 3
    topo[m1].inqueues[0].append(r1)
    topo[m1].inqueues[0].append(r2)
    _inject_inflight(m2, [0])

    assert _global_min_epoch(topo, microbatch_size=B) == 0


def test_global_min_epoch_returns_none_when_no_stamped_bundles():
    topo, _ = _linear_topology(num_ops=1)
    assert _global_min_epoch(topo, microbatch_size=1) is None


def test_microbatch_select_prefers_shallowest_in_active_epoch():
    topo, ops = _linear_topology(num_ops=2)
    _, m1, m2 = ops
    # Both m1 and m2 have epoch-0 work ready. Under strict BSP, the stage
    # barrier blocks m2 (m1 has epoch-0 work queued), so m1 is the sole
    # eligible op.
    r_m1 = make_ref_bundles([["a"]])[0]
    r_m1.partition_index = 1  # epoch 0 with B=4
    r_m2 = make_ref_bundles([["b"]])[0]
    r_m2.partition_index = 0  # epoch 0 with B=4
    topo[m1].inqueues[0].append(r_m1)
    topo[m2].inqueues[0].append(r_m2)

    selected = _microbatch_select([m1, m2], topo, microbatch_size=4)
    assert selected is m1


def test_microbatch_select_blocks_on_later_epoch_when_earlier_in_flight():
    topo, ops = _linear_topology(num_ops=2)
    _, m1, m2 = ops
    # m2 is still processing epoch 0 (in-flight).
    _inject_inflight(m2, [1])  # epoch 0 with B=4
    # m1 has epoch-1 work queued (partition_index 4).
    r_m1 = make_ref_bundles([["a"]])[0]
    r_m1.partition_index = 4
    topo[m1].inqueues[0].append(r_m1)

    # m1 cannot run: inter-batch barrier forbids epoch 1 until epoch 0 drains.
    selected = _microbatch_select([m1], topo, microbatch_size=4)
    assert selected is None


def test_microbatch_select_advances_to_next_epoch_after_drain():
    topo, ops = _linear_topology(num_ops=1)
    _, m1 = ops
    # Only epoch-1 work exists; should select m1 (no earlier work anywhere).
    r = make_ref_bundles([["a"]])[0]
    r.partition_index = 5  # epoch 1 with B=4
    topo[m1].inqueues[0].append(r)

    selected = _microbatch_select([m1], topo, microbatch_size=4)
    assert selected is m1


def test_microbatch_select_falls_back_when_bundles_unstamped():
    topo, ops = _linear_topology(num_ops=1)
    _, m1 = ops
    # Unstamped bundle in m1's inqueue.
    r = make_ref_bundles([["a"]])[0]
    assert r.partition_index is None
    topo[m1].inqueues[0].append(r)

    selected = _microbatch_select([m1], topo, microbatch_size=4)
    assert selected is m1


def test_transitive_ancestors_walks_dag():
    topo, ops = _linear_topology(num_ops=3)
    src, m1, m2, m3 = ops
    assert set(_transitive_ancestors(m3)) == {m2, m1, src}
    assert set(_transitive_ancestors(m1)) == {src}
    assert _transitive_ancestors(src) == []


def test_op_has_pending_at_epoch_checks_inflight_and_inqueue_head():
    topo, ops = _linear_topology(num_ops=1)
    _, m1 = ops
    state = topo[m1]
    assert not _op_has_pending_at_epoch(state, 0, microbatch_size=2)

    # in-flight p=1 => epoch 0
    _inject_inflight(m1, [1])
    assert _op_has_pending_at_epoch(state, 0, microbatch_size=2)
    assert not _op_has_pending_at_epoch(state, 1, microbatch_size=2)
    m1._data_tasks.clear()

    # queued head p=3 => epoch 1
    r = make_ref_bundles([["a"]])[0]
    r.partition_index = 3
    state.inqueues[0].append(r)
    assert not _op_has_pending_at_epoch(state, 0, microbatch_size=2)
    assert _op_has_pending_at_epoch(state, 1, microbatch_size=2)


def test_microbatch_select_stage_barrier_blocks_on_inflight_upstream():
    """classify cannot start epoch 0 while map still has epoch-0 in-flight."""
    topo, ops = _linear_topology(num_ops=2)
    _, map_op, classify = ops
    # map is running epoch-0 work.
    _inject_inflight(map_op, [0])  # epoch 0 with B=2
    # classify has an epoch-0 bundle ready.
    r = make_ref_bundles([["a"]])[0]
    r.partition_index = 0
    topo[classify].inqueues[0].append(r)

    selected = _microbatch_select([classify], topo, microbatch_size=2)
    assert selected is None


def test_microbatch_select_stage_barrier_blocks_on_queued_upstream():
    """classify cannot start epoch 0 while map still has epoch-0 queued."""
    topo, ops = _linear_topology(num_ops=2)
    _, map_op, classify = ops
    # map has epoch-0 work queued (head at epoch 0).
    r_map = make_ref_bundles([["a"]])[0]
    r_map.partition_index = 1  # epoch 0 with B=2
    topo[map_op].inqueues[0].append(r_map)
    # classify has an epoch-0 bundle ready too.
    r_cls = make_ref_bundles([["b"]])[0]
    r_cls.partition_index = 0
    topo[classify].inqueues[0].append(r_cls)

    # Under strict BSP, with both map and classify eligible for epoch 0,
    # map should win (shallower) because the stage barrier blocks classify.
    selected = _microbatch_select([map_op, classify], topo, microbatch_size=2)
    assert selected is map_op


def test_microbatch_select_stage_barrier_passes_when_upstream_drained():
    """classify may run epoch 0 once map has no epoch-0 work anywhere."""
    topo, ops = _linear_topology(num_ops=2)
    _, map_op, classify = ops
    # map has only epoch-1 work (no epoch-0 in-flight or queued).
    r_map = make_ref_bundles([["a"]])[0]
    r_map.partition_index = 2  # epoch 1 with B=2
    topo[map_op].inqueues[0].append(r_map)
    # classify has epoch-0 ready.
    r_cls = make_ref_bundles([["b"]])[0]
    r_cls.partition_index = 0
    topo[classify].inqueues[0].append(r_cls)

    # min_e = 0 (classify's head), map's head is epoch 1 so not eligible.
    # classify's only ancestor with state is map, which has no epoch-0 work.
    selected = _microbatch_select([classify], topo, microbatch_size=2)
    assert selected is classify


def test_microbatch_select_stage_barrier_traverses_transitively():
    """With read -> map -> classify, epoch-0 work on `read` also blocks classify."""
    topo, ops = _linear_topology(num_ops=3)
    _, read_op, map_op, classify = ops
    # read is still running epoch-0 work (in-flight).
    _inject_inflight(read_op, [1])  # epoch 0 with B=2
    # classify has epoch-0 ready (synthetic shortcut for the test).
    r = make_ref_bundles([["a"]])[0]
    r.partition_index = 0
    topo[classify].inqueues[0].append(r)

    # classify blocked: read (transitive ancestor) still has epoch-0 work.
    selected = _microbatch_select([classify], topo, microbatch_size=2)
    assert selected is None


# ---- Drizzle toggles: group_size (G) and stage_barrier="relaxed" ----


def test_group_scheduling_admits_epochs_within_window():
    """With G=2 and B=2, a head at epoch 0 is inside the window [0, 2)."""
    topo, ops = _linear_topology(num_ops=1)
    _, m1 = ops
    r0 = make_ref_bundles([["a"]])[0]
    r0.partition_index = 0  # epoch 0
    topo[m1].inqueues[0].append(r0)

    selected = _microbatch_select(
        [m1], topo, microbatch_size=2, group_size=2, stage_barrier="strict"
    )
    assert selected is m1


def test_group_scheduling_blocks_epoch_outside_window():
    """min_e=0, G=2 => epoch 2 head is outside [0, 2) and must wait."""
    topo, ops = _linear_topology(num_ops=2)
    _, m1, m2 = ops
    # m2 still running epoch-0 work, holds min_e at 0.
    _inject_inflight(m2, [0])  # epoch 0
    # m1 has epoch-2 work queued (pid 4 with B=2).
    r = make_ref_bundles([["a"]])[0]
    r.partition_index = 4
    topo[m1].inqueues[0].append(r)

    selected = _microbatch_select(
        [m1], topo, microbatch_size=2, group_size=2, stage_barrier="strict"
    )
    assert selected is None  # epoch 2 outside [0, 2)

    # But with G=3, the window [0, 3) admits epoch 2.
    selected = _microbatch_select(
        [m1], topo, microbatch_size=2, group_size=3, stage_barrier="strict"
    )
    assert selected is m1


def test_prescheduling_allows_downstream_while_upstream_busy():
    """With stage_barrier="relaxed", classify may dispatch epoch-0 even while
    map still has epoch-0 in-flight (pure Drizzle pre-scheduling)."""
    topo, ops = _linear_topology(num_ops=2)
    _, map_op, classify = ops
    # map has epoch-0 work in flight (would block classify under strict).
    _inject_inflight(map_op, [1])
    # classify has epoch-0 bundle ready.
    r = make_ref_bundles([["a"]])[0]
    r.partition_index = 0
    topo[classify].inqueues[0].append(r)

    # Strict barrier: blocked.
    assert (
        _microbatch_select(
            [classify], topo, microbatch_size=2, stage_barrier="strict"
        )
        is None
    )
    # Relaxed barrier: dispatches.
    assert (
        _microbatch_select(
            [classify], topo, microbatch_size=2, stage_barrier="relaxed"
        )
        is classify
    )


def test_drizzle_full_allows_multi_epoch_and_prescheduling():
    """Full Drizzle (G=2, relaxed): classify may start epoch 0 while map
    processes epoch 1, and epoch 1 is admissible via the G=2 window."""
    topo, ops = _linear_topology(num_ops=2)
    _, map_op, classify = ops
    # map has epoch-1 work (pid 2) queued — outside strict-BSP min_e=0, but
    # allowed under G=2.
    r_map = make_ref_bundles([["a"]])[0]
    r_map.partition_index = 2
    topo[map_op].inqueues[0].append(r_map)
    # classify has epoch-0 work (pid 0). Under strict stage barrier, classify's
    # ancestor (map) has no epoch-0 work so it passes; under relaxed, skipped.
    r_cls = make_ref_bundles([["b"]])[0]
    r_cls.partition_index = 0
    topo[classify].inqueues[0].append(r_cls)

    # With G=2 + relaxed, both map (epoch 1 in window) and classify (epoch 0)
    # are eligible. Shallowest (map) wins.
    selected = _microbatch_select(
        [map_op, classify], topo, microbatch_size=2,
        group_size=2, stage_barrier="relaxed",
    )
    assert selected is map_op


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
