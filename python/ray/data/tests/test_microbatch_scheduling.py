"""Unit tests for the microbatch (BSP) scheduling policy.

Covers:
  - InputDataBuffer stamps partition_index on bundles.
  - OpState propagates partition_index through dispatch/add_output.
  - The microbatch helpers `_compute_depth_map` and `_global_min_epoch`.
  - `_microbatch_select` enforces intra-batch stage and inter-batch barriers.
"""

from unittest.mock import MagicMock

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
    OpState,
    _compute_depth_map,
    _global_min_epoch,
    _microbatch_select,
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


def test_add_output_propagates_partition_index():
    topo, ops = _linear_topology(num_ops=1)
    src, m1 = ops
    # Simulate a bundle having been dispatched into m1 with partition_index=5.
    state = topo[m1]
    state._inflight_partition_indices.append(5)

    out_bundle = make_ref_bundles([["out"]])[0]
    assert out_bundle.partition_index is None
    state.add_output(out_bundle)
    assert out_bundle.partition_index == 5
    assert not state._inflight_partition_indices


def test_dispatch_next_task_tracks_partition_index():
    inputs = make_ref_bundles([[x] for x in range(2)])
    src = InputDataBuffer(inputs)
    src_state = OpState(src, [])
    m1 = MapOperator.create(_make_map_transformer(lambda b: b), src)
    m1_state = OpState(m1, [src_state.outqueue])

    # Stamp the bundle and feed it via the shared inqueue/outqueue.
    ref = make_ref_bundles([["x"]])[0]
    ref.partition_index = 7
    m1_state.inqueues[0].append(ref)

    m1.add_input = MagicMock()
    m1_state.dispatch_next_task()
    m1.add_input.assert_called_once_with(ref, input_index=0)
    assert list(m1_state._inflight_partition_indices) == [7]


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
    topo[m2]._inflight_partition_indices.append(0)

    assert _global_min_epoch(topo, microbatch_size=B) == 0


def test_global_min_epoch_returns_none_when_no_stamped_bundles():
    topo, _ = _linear_topology(num_ops=1)
    assert _global_min_epoch(topo, microbatch_size=1) is None


def test_microbatch_select_prefers_shallowest_in_active_epoch():
    topo, ops = _linear_topology(num_ops=2)
    _, m1, m2 = ops
    # Both m1 and m2 have epoch-0 work ready; scheduler should pick m1 (shallower).
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
    topo[m2]._inflight_partition_indices.append(1)  # epoch 0 with B=4
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
