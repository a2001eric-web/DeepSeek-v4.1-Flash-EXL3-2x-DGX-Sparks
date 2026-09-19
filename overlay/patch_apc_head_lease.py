#!/usr/bin/env python3
"""Conversation-scoped prefix-cache head leases for the shared DSV4.1 pool.

The stock vLLM block pool is global LRU.  Interleaved long conversations can
therefore evict another conversation's latest reusable prefix while older,
less useful branches remain cached.  This overlay adds a best-effort eviction
priority tier, not a reservation:

* V2 sends opaque ``prefix_cache_lease_scope`` and ``prefix_cache_lease``
  values through ``vllm_xargs``.
* Only a normally stopped request replaces that lease's protected head.
* Starting a new session in the same scope releases the previous head.
* Unhashed and unleased cached blocks are recycled before leased heads.
* Under real capacity pressure leased blocks are still evicted normally.

No refcount is raised and no block is removed from the free queue, so admission
capacity is unchanged.  Malformed wire values fail at the API boundary, while
aborted, errored, ignored, length-capped, and repetition-stopped requests never
replace a known-good head.  The strict ``DSV41_APC_HEAD_LEASE=0`` kill switch
disables ownership at engine ingress.  Patching is transactional, idempotent,
and fail-closed against an unknown vLLM source shape.
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

MARK = "# [dsv41-apc-head-lease]"
_VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
SAMPLING_PARAMS_PY = Path(
    os.environ.get("DSV41_SAMPLING_PARAMS_PY", f"{_VLLM}/sampling_params.py")
)
REQUEST_PY = Path(os.environ.get("DSV41_REQUEST_PY", f"{_VLLM}/v1/request.py"))
BLOCK_POOL_PY = Path(
    os.environ.get("DSV41_BLOCK_POOL_PY", f"{_VLLM}/v1/core/block_pool.py")
)
KV_MANAGER_PY = Path(
    os.environ.get(
        "DSV41_KV_MANAGER_PY", f"{_VLLM}/v1/core/kv_cache_manager.py"
    )
)


# sampling_params.py ---------------------------------------------------------

SP_CLASS_ANCHOR = "\nclass SamplingParams(\n"
SP_HELPERS = r'''
# [dsv41-apc-head-lease] helper-begin
_DSV41_HEAD_SCOPE_KEY = "prefix_cache_lease_scope"
_DSV41_HEAD_LEASE_KEY = "prefix_cache_lease"
_DSV41_HEAD_LEASE_ENV = "DSV41_APC_HEAD_LEASE"


def _dsv41_head_lease_enabled():  # [dsv41-apc-head-lease]
    import os as _os

    raw = _os.environ.get(_DSV41_HEAD_LEASE_ENV)
    if raw is None or raw == "1":
        return True
    if raw == "0":
        return False
    raise ValueError(
        f"{_DSV41_HEAD_LEASE_ENV} must be exactly 0 or 1 (got {raw!r}); "
        "unset it to keep conversation head leases enabled."
    )


_DSV41_HEAD_LEASE_ON = _dsv41_head_lease_enabled()


def _dsv41_parse_head_token(value, where):  # [dsv41-apc-head-lease]
    """Strict bounded ASCII token; identities must already be opaque."""
    import re as _re

    if not isinstance(value, str) or not _re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value
    ):
        raise ValueError(
            f"{where} must be a 1..128 character ASCII token using only "
            "letters, digits, '.', '_', ':' or '-'; got {value!r}."
        )
    return value


def _dsv41_validate_head_lease_params(params):  # [dsv41-apc-head-lease]
    extra = params.extra_args or {}
    has_scope = _DSV41_HEAD_SCOPE_KEY in extra
    has_lease = _DSV41_HEAD_LEASE_KEY in extra
    if has_scope != has_lease:
        raise ValueError(
            "vllm_xargs prefix_cache_lease_scope and prefix_cache_lease "
            "must be supplied together."
        )
    if has_scope:
        _dsv41_parse_head_token(
            extra[_DSV41_HEAD_SCOPE_KEY],
            'vllm_xargs["prefix_cache_lease_scope"]',
        )
        _dsv41_parse_head_token(
            extra[_DSV41_HEAD_LEASE_KEY],
            'vllm_xargs["prefix_cache_lease"]',
        )


def _dsv41_resolve_head_lease(sampling_params, request_id):
    """Resolve validated request metadata once at engine ingress."""
    if sampling_params is None:
        return None, None
    extra = getattr(sampling_params, "extra_args", None) or {}
    if not ({_DSV41_HEAD_SCOPE_KEY, _DSV41_HEAD_LEASE_KEY} <= extra.keys()):
        return None, None
    try:
        scope = _dsv41_parse_head_token(
            extra[_DSV41_HEAD_SCOPE_KEY],
            'vllm_xargs["prefix_cache_lease_scope"]',
        )
        lease = _dsv41_parse_head_token(
            extra[_DSV41_HEAD_LEASE_KEY],
            'vllm_xargs["prefix_cache_lease"]',
        )
    except ValueError as exc:
        logger.warning(
            "[dsv41-apc-head-lease] %s -- request %s remains unleased "
            "(this should have been rejected at the API boundary)",
            exc,
            request_id,
        )
        return None, None
    if not _DSV41_HEAD_LEASE_ON:
        logger.info_once(
            "[dsv41-apc-head-lease] valid lease ignored: "
            "DSV41_APC_HEAD_LEASE=0"
        )
        return None, None
    logger.info_once(
        "[dsv41-apc-head-lease] first conversation lease reached engine core"
    )
    return scope, lease
# [dsv41-apc-head-lease] helper-end

'''

SP_POST_OLD = "        self._verify_args()\n"
SP_POST_NEW = SP_POST_OLD + (
    "        _dsv41_validate_head_lease_params(self)  "
    "# [dsv41-apc-head-lease]\n"
)


# request.py -----------------------------------------------------------------

REQ_RESOLVE_OLD = "        self.stop_reason: int | str | None = None\n"
REQ_RESOLVE_NEW = REQ_RESOLVE_OLD + '''

        # [dsv41-apc-head-lease] opaque owner generation; resolved once
        from vllm.sampling_params import _dsv41_resolve_head_lease
        (
            self.prefix_cache_lease_scope,
            self.prefix_cache_lease,
        ) = _dsv41_resolve_head_lease(self.sampling_params, self.request_id)
'''


# block_pool.py --------------------------------------------------------------

BP_INIT_OLD = "        self.metrics_collector = metrics_collector\n"
BP_INIT_NEW = BP_INIT_OLD + '''

        # [dsv41-apc-head-lease] Best-effort priority metadata only. Leased
        # blocks stay ref_cnt=0 and remain allocatable under real pressure.
        self._dsv41_lease_heads: dict[str, set[int]] = {}
        self._dsv41_block_head_leases: dict[int, set[str]] = {}
        self._dsv41_scope_lease: dict[str, str] = {}
        self._dsv41_head_pressure_evictions = 0
'''

BP_GET_ANCHOR = "    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:\n"
BP_METHODS = r'''    def _dsv41_free_queue_reposition(self, block, *, protected):  # [dsv41-apc-head-lease]
        """Keep unprotected candidates before leased heads without pinning."""
        if (
            block.is_null
            or block.ref_cnt != 0
            or block.prev_free_block is None
            or block.next_free_block is None
        ):
            return
        self.free_block_queue.remove(block)
        if protected:
            self.free_block_queue.append(block)
        else:
            self.free_block_queue.prepend_n([block])

    def _dsv41_forget_head_block(self, block_id):  # [dsv41-apc-head-lease]
        owners = self._dsv41_block_head_leases.pop(block_id, None)
        if not owners:
            return
        for lease in owners:
            head = self._dsv41_lease_heads.get(lease)
            if head is not None:
                head.discard(block_id)

    def _dsv41_drop_head_lease(self, lease):  # [dsv41-apc-head-lease]
        old_ids = self._dsv41_lease_heads.pop(lease, set())
        for block_id in old_ids:
            owners = self._dsv41_block_head_leases.get(block_id)
            if owners is None:
                continue
            owners.discard(lease)
            if owners:
                continue
            self._dsv41_block_head_leases.pop(block_id, None)
            self._dsv41_free_queue_reposition(
                self.blocks[block_id], protected=False
            )

    def _dsv41_transfer_head_block(self, src_block, dst_block):
        """Follow partial-hit CoW hash relocation without losing ownership."""
        owners = self._dsv41_block_head_leases.pop(src_block.block_id, None)
        if not owners:
            return
        dst_owners = self._dsv41_block_head_leases.setdefault(
            dst_block.block_id, set()
        )
        dst_owners.update(owners)
        for lease in owners:
            head = self._dsv41_lease_heads.setdefault(lease, set())
            head.discard(src_block.block_id)
            head.add(dst_block.block_id)
        self._dsv41_free_queue_reposition(dst_block, protected=True)

    def commit_prefix_cache_head(self, scope, lease, blocks):
        """Replace one conversation's latest reusable head.

        This changes eviction order only. It never changes block refcounts,
        cache hashes, admission capacity, or hit eligibility.
        """
        if not self.enable_caching or not scope or not lease:
            return

        previous = self._dsv41_scope_lease.pop(scope, None)
        self._dsv41_scope_lease[scope] = lease
        if previous and previous != lease:
            if previous not in self._dsv41_scope_lease.values():
                self._dsv41_drop_head_lease(previous)
            logger.info(
                "[dsv41-apc-head-lease] scope rotated; old head released"
            )

        # Bound inactive scope metadata. Cached blocks remain bounded by the
        # physical pool; this only caps tiny ownership maps.
        while len(self._dsv41_scope_lease) > 64:
            old_scope = next(iter(self._dsv41_scope_lease))
            old_lease = self._dsv41_scope_lease.pop(old_scope)
            if old_lease not in self._dsv41_scope_lease.values():
                self._dsv41_drop_head_lease(old_lease)

        new_ids = {
            block.block_id
            for block in blocks
            if not block.is_null and block.block_hash is not None
        }
        old_ids = self._dsv41_lease_heads.get(lease, set())
        for block_id in old_ids - new_ids:
            owners = self._dsv41_block_head_leases.get(block_id)
            if owners is None:
                continue
            owners.discard(lease)
            if not owners:
                self._dsv41_block_head_leases.pop(block_id, None)
                self._dsv41_free_queue_reposition(
                    self.blocks[block_id], protected=False
                )
        for block_id in new_ids - old_ids:
            self._dsv41_block_head_leases.setdefault(block_id, set()).add(lease)
            self._dsv41_free_queue_reposition(
                self.blocks[block_id], protected=True
            )
        self._dsv41_lease_heads[lease] = new_ids
        logger.debug(
            "[dsv41-apc-head-lease] committed head blocks=%d leases=%d scopes=%d",
            len(new_ids),
            len(self._dsv41_lease_heads),
            len(self._dsv41_scope_lease),
        )

'''

BP_MOVE_OLD = '''        for block_hash in self._remove_cached_block_hashes(src_block):
            # `num_tokens` only applies to the first (primary) insertion.
            self._insert_block_hash(block_hash, dst_block, num_tokens=num_tokens)
'''
BP_MOVE_NEW = BP_MOVE_OLD + '''        self._dsv41_transfer_head_block(  # [dsv41-apc-head-lease]
            src_block, dst_block
        )
'''

BP_POP_OLD = "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)\n"
BP_POP_NEW = BP_POP_OLD + '''        protected = sum(
            block.block_id in self._dsv41_block_head_leases for block in ret
        )
        if protected:
            self._dsv41_head_pressure_evictions += protected
            logger.warning_once(
                "[dsv41-apc-head-lease] capacity pressure reached leased "
                "conversation heads; eviction remains allowed (first occurrence)"
            )
            logger.debug(
                "[dsv41-apc-head-lease] pressure evicted=%d cumulative=%d",
                protected,
                self._dsv41_head_pressure_evictions,
            )
'''

BP_EVICT_OLD = '''        self._emit_block_removed_events(evicted_hashes)
        return True
'''
BP_EVICT_NEW = '''        self._emit_block_removed_events(evicted_hashes)
        self._dsv41_forget_head_block(  # [dsv41-apc-head-lease]
            block.block_id
        )
        return True
'''

BP_FREE_DECL_OLD = '''        blocks_to_evict_last = []
        blocks_to_evict_first = []
'''
BP_FREE_DECL_NEW = '''        blocks_to_evict_last = []
        blocks_to_evict_first = []
        blocks_to_evict_before_leased_heads = []  # [dsv41-apc-head-lease]
        blocks_to_evict_as_leased_heads = []
'''

BP_FREE_NORMAL_OLD = '''                else:
                    # FIFO reuse of cached blocks for LRU eviction behavior.
                    blocks_to_evict_last.append(block)
'''
BP_FREE_NORMAL_NEW = '''                else:
                    # A leased head remains at the tail; ordinary cached state
                    # stays ahead of it and is recycled first.
                    if block.block_id in self._dsv41_block_head_leases:
                        blocks_to_evict_as_leased_heads.append(block)
                    elif self._dsv41_block_head_leases:
                        blocks_to_evict_before_leased_heads.append(block)
                    else:
                        # Stock FIFO/LRU path when no lease is active.
                        blocks_to_evict_last.append(block)
'''

BP_FREE_TAIL_OLD = '''        self.free_block_queue.prepend_n(blocks_to_evict_first)
        # Blocks to reuse last are appended to the end of the free queue.
        self.free_block_queue.append_n(blocks_to_evict_last)
'''
BP_FREE_TAIL_NEW = '''        self.free_block_queue.prepend_n(blocks_to_evict_before_leased_heads)
        self.free_block_queue.prepend_n(blocks_to_evict_first)
        # Blocks to reuse last are appended to the end of the free queue.
        self.free_block_queue.append_n(blocks_to_evict_last)
        self.free_block_queue.append_n(blocks_to_evict_as_leased_heads)
'''

BP_RESET_OLD = '''        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.cached_block_hashes_by_block.clear()
'''
BP_RESET_NEW = BP_RESET_OLD + '''        self._dsv41_lease_heads.clear()  # [dsv41-apc-head-lease]
        self._dsv41_block_head_leases.clear()
        self._dsv41_scope_lease.clear()
        self._dsv41_head_pressure_evictions = 0
'''


# kv_cache_manager.py --------------------------------------------------------

KVM_CLASS_ANCHOR = "\nclass KVCacheManager:\n"
KVM_METHOD = r'''    def _dsv41_commit_prefix_cache_head(self, request):  # [dsv41-apc-head-lease]
        """Commit only a normally completed request before bookkeeping is popped."""
        if request.status != RequestStatus.FINISHED_STOPPED:
            return
        scope = getattr(request, "prefix_cache_lease_scope", None)
        lease = getattr(request, "prefix_cache_lease", None)
        if not scope or not lease:
            return
        blocks = []
        for manager in self.coordinator.single_type_managers:
            blocks.extend(manager.req_to_blocks.get(request.request_id, ()))
        self.block_pool.commit_prefix_cache_head(scope, lease, blocks)

'''

KVM_FREE_OLD = "        self.coordinator.free(request.request_id)\n"
KVM_FREE_NEW = '''        self._dsv41_commit_prefix_cache_head(  # [dsv41-apc-head-lease]
            request
        )
        self.coordinator.free(request.request_id)
'''

KVM_POP_OLD = "        return self.coordinator.pop_blocks_for_free(request.request_id)\n"
KVM_POP_NEW = '''        self._dsv41_commit_prefix_cache_head(  # [dsv41-apc-head-lease]
            request
        )
        return self.coordinator.pop_blocks_for_free(request.request_id)
'''


PLAN = {
    "sampling_params.py": (
        SAMPLING_PARAMS_PY,
        (
            ("sampling-helpers", SP_CLASS_ANCHOR, SP_HELPERS + SP_CLASS_ANCHOR),
            ("sampling-validation", SP_POST_OLD, SP_POST_NEW),
        ),
        (),
    ),
    "request.py": (
        REQUEST_PY,
        (("request-resolve", REQ_RESOLVE_OLD, REQ_RESOLVE_NEW),),
        (),
    ),
    "block_pool.py": (
        BLOCK_POOL_PY,
        (
            ("block-init", BP_INIT_OLD, BP_INIT_NEW),
            ("block-methods", BP_GET_ANCHOR, BP_METHODS + BP_GET_ANCHOR),
            ("block-move", BP_MOVE_OLD, BP_MOVE_NEW),
            ("block-pop", BP_POP_OLD, BP_POP_NEW),
            ("block-evict", BP_EVICT_OLD, BP_EVICT_NEW),
            ("block-free-decl", BP_FREE_DECL_OLD, BP_FREE_DECL_NEW),
            ("block-free-normal", BP_FREE_NORMAL_OLD, BP_FREE_NORMAL_NEW),
            ("block-free-tail", BP_FREE_TAIL_OLD, BP_FREE_TAIL_NEW),
            ("block-reset", BP_RESET_OLD, BP_RESET_NEW),
        ),
        (),
    ),
    "kv_cache_manager.py": (
        KV_MANAGER_PY,
        (
            ("manager-method", KVM_CLASS_ANCHOR, KVM_CLASS_ANCHOR + KVM_METHOD),
            ("manager-free", KVM_FREE_OLD, KVM_FREE_NEW),
            ("manager-pop", KVM_POP_OLD, KVM_POP_NEW),
        ),
        ("from vllm.v1.request import Request, RequestStatus",),
    ),
}


def _expected_marks(edits) -> int:
    return sum(new.count(MARK) - old.count(MARK) for _, old, new in edits)


def _parse(text: str, path: Path) -> None:
    try:
        ast.parse(text, str(path))
    except SyntaxError as exc:
        raise SystemExit(f"{path}: patched source does not parse: {exc}") from None


def _preflight(name: str, path: Path, edits, requires) -> str | None:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    text = path.read_text()
    have = text.count(MARK)
    want = _expected_marks(edits)
    if have:
        missing = [label for label, _old, new in edits if text.count(new) != 1]
        if have != want or missing:
            raise SystemExit(
                f"{path}: incomplete {MARK} overlay "
                f"(markers={have}, expected={want}, missing={missing})"
            )
        _parse(text, path)
        return None
    for required in requires:
        if required not in text:
            raise SystemExit(f"{path}: prerequisite missing: {required!r}")
    for label, old, _new in edits:
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"{path}: anchor {label!r} occurs {count} times (expected 1)"
            )
    patched = text
    for _label, old, new in edits:
        patched = patched.replace(old, new, 1)
    if patched.count(MARK) != want:
        raise SystemExit(f"{path}: marker count mismatch after patch")
    _parse(patched, path)
    return patched


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    staged: list[tuple[Path, str]] = []
    for name, (path, edits, requires) in PLAN.items():
        patched = _preflight(name, path, edits, requires)
        if patched is not None:
            staged.append((path, patched))
    for path, text in staged:
        _atomic_write(path, text)
    print(
        "dsv41: APC conversation head lease overlay "
        f"{'installed' if staged else 'already present'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
