#!/usr/bin/env python3
"""CPU-only contract tests for the DSV4.1 APC head-lease overlay."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
PATCH = next(
    path
    for path in (
        ROOT / "overlay" / "patch_apc_head_lease.py",
        HERE / "patch_apc_head_lease.py",
    )
    if path.is_file()
)
START = ROOT / "start.sh"
DOCKERFILE = ROOT / "Dockerfile"


def test_recipe_wiring() -> None:
    ast.parse(PATCH.read_text(), str(PATCH))
    if not START.is_file() or not DOCKERFILE.is_file():
        return
    start = START.read_text()
    dockerfile = DOCKERFILE.read_text()
    assert start.count("python3 /opt/dsv41/patch_apc_head_lease.py") == 2
    assert start.count(":/opt/dsv41/patch_apc_head_lease.py:ro") == 2
    assert 'DSV41_APC_HEAD_LEASE="${DSV41_APC_HEAD_LEASE:-1}"' in start
    assert "_glm53_validate_enum DSV41_APC_HEAD_LEASE" in start
    assert "RUN python3 /opt/dsv41/patch_apc_head_lease.py" in dockerfile
    assert "DSV41_REQUIRE_VLLM=1 python3 /opt/dsv41/test_apc_head_lease.py" in dockerfile


def _runtime_available() -> bool:
    try:
        import vllm.sampling_params  # noqa: F401

        from vllm.v1.core.block_pool import BlockPool  # noqa: F401

        return True
    except ImportError:
        if os.environ.get("DSV41_REQUIRE_VLLM") == "1":
            raise
        return False


def test_sampling_contract() -> None:
    if not _runtime_available():
        return

    from vllm.sampling_params import SamplingParams

    good = {
        "prefix_cache_lease_scope": "scope-v1:0123456789abcdef",
        "prefix_cache_lease": "session-v1:fedcba9876543210",
    }
    SamplingParams(max_tokens=1, extra_args=good)

    bad_values = ("", "has space", "../escape", "x" * 129, 7, None)
    for value in bad_values:
        try:
            SamplingParams(
                max_tokens=1,
                extra_args={**good, "prefix_cache_lease": value},
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed lease was accepted: {value!r}")

    for one_sided in (
        {"prefix_cache_lease_scope": "scope-v1:a"},
        {"prefix_cache_lease": "session-v1:a"},
    ):
        try:
            SamplingParams(max_tokens=1, extra_args=one_sided)
        except ValueError:
            pass
        else:
            raise AssertionError(f"one-sided lease was accepted: {one_sided!r}")


def _queue_ids(pool) -> list[int]:
    ids: list[int] = []
    block = pool.free_block_queue.fake_free_list_head.next_free_block
    tail = pool.free_block_queue.fake_free_list_tail
    while block is not tail:
        assert block is not None
        ids.append(block.block_id)
        block = block.next_free_block
    return ids


def _hash(pool, block, seed: int) -> None:
    from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

    key = make_block_hash_with_group_id(BlockHash(bytes([seed]) * 32), 0)
    pool._insert_block_hash(key, block, num_tokens=64)


def test_priority_is_best_effort_not_a_pin() -> None:
    if not _runtime_available():
        return

    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(num_gpu_blocks=10, enable_caching=True, hash_block_size=64)

    first = pool.get_new_blocks(3)
    for index, block in enumerate(first, 1):
        _hash(pool, block, index)
    pool.commit_prefix_cache_head("scope:a", "lease:a1", first)
    pool.free_blocks(first)

    queue = _queue_ids(pool)
    protected = {block.block_id for block in first}
    assert set(queue[-3:]) == protected
    assert all(pool.blocks[block_id].ref_cnt == 0 for block_id in protected)

    second = pool.get_new_blocks(2)
    assert protected.isdisjoint(block.block_id for block in second)
    for index, block in enumerate(second, 10):
        _hash(pool, block, index)
    pool.commit_prefix_cache_head("scope:b", "lease:b1", second)
    pool.free_blocks(second)

    # A new generation for scope:a releases lease:a1 immediately.
    replacement = next(
        block
        for block in pool.blocks
        if not block.is_null
        and block.ref_cnt == 0
        and block.block_id not in protected
        and block.block_id not in {item.block_id for item in second}
    )
    _hash(pool, replacement, 20)
    pool.commit_prefix_cache_head("scope:a", "lease:a2", [replacement])
    assert "lease:a1" not in pool._dsv41_lease_heads
    assert replacement.block_id in pool._dsv41_block_head_leases

    # Real pressure still drains the whole queue, including protected heads.
    allocated = pool.get_new_blocks(pool.get_num_free_blocks())
    assert pool._dsv41_head_pressure_evictions > 0
    assert {item.block_id for item in allocated} >= {
        replacement.block_id,
        *(item.block_id for item in second),
    }
    pool.free_blocks(allocated)


def test_only_normal_stop_commits() -> None:
    if not _runtime_available():
        return

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import RequestStatus

    calls: list[tuple[str, str, list[object]]] = []
    blocks = [object(), object()]
    fake_manager = SimpleNamespace(req_to_blocks={"req": blocks})
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.coordinator = SimpleNamespace(single_type_managers=[fake_manager])
    manager.block_pool = SimpleNamespace(
        commit_prefix_cache_head=lambda scope, lease, values: calls.append(
            (scope, lease, list(values))
        )
    )

    for status in (
        RequestStatus.FINISHED_ABORTED,
        RequestStatus.FINISHED_LENGTH_CAPPED,
        RequestStatus.FINISHED_IGNORED,
        RequestStatus.FINISHED_ERROR,
        RequestStatus.FINISHED_REPETITION,
    ):
        request = SimpleNamespace(
            request_id="req",
            status=status,
            prefix_cache_lease_scope="scope:a",
            prefix_cache_lease="lease:a1",
        )
        manager._dsv41_commit_prefix_cache_head(request)
    assert calls == []

    request.status = RequestStatus.FINISHED_STOPPED
    manager._dsv41_commit_prefix_cache_head(request)
    assert calls == [("scope:a", "lease:a1", blocks)]


def test_reset_clears_lease_metadata() -> None:
    if not _runtime_available():
        return

    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=64)
    block = pool.get_new_blocks(1)[0]
    _hash(pool, block, 30)
    pool.commit_prefix_cache_head("scope:a", "lease:a", [block])
    pool.free_blocks([block])
    assert pool._dsv41_lease_heads
    assert pool.reset_prefix_cache() is True
    assert pool._dsv41_lease_heads == {}
    assert pool._dsv41_block_head_leases == {}
    assert pool._dsv41_scope_lease == {}


def main() -> None:
    test_recipe_wiring()
    test_sampling_contract()
    test_priority_is_best_effort_not_a_pin()
    test_only_normal_stop_commits()
    test_reset_clears_lease_metadata()
    print("APC head lease tests: PASS")


if __name__ == "__main__":
    main()
