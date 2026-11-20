"""存储层面：分叉共享 KV 视图、追加只写新段、可见引用回收、截断分叉、超限原子拒绝。"""

import threading

import numpy as np
import pytest

from app.model import build_model, forward_full
from app.store import BranchStore, CapacityExceeded


MODEL = build_model()


def _assert_logits_match(store, branch):
    logits = store._logits_at(branch)
    full = forward_full(MODEL, branch.token_ids)
    np.testing.assert_allclose(logits, full, rtol=1e-5, atol=1e-6)


def _assert_invariants(store):
    snap = store.debug_snapshot()
    assert snap["occupied_matches_live_kv"]
    assert snap["occupied_matches_stored_rows"]
    for b in snap["branches"]:
        assert b["view_length"] <= b["kv_length"]
        if b["kv_segments"]:
            assert b["kv_refcount"] >= 1
    assert snap["usage"]["occupied_tokens"] <= snap["usage"]["max_cached_tokens"]


def test_create_context_and_usage():
    store = BranchStore(MODEL, max_cached_tokens=128)
    branch, logits = store.create_branch([1, 2, 3, 4])
    assert branch.token_ids == [1, 2, 3, 4]
    assert store.usage()["occupied_tokens"] == 4
    _assert_logits_match(store, branch)
    _assert_invariants(store)


def test_fork_shares_kv_without_extra_occupancy():
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([1, 2, 3, 4])
    occupied_before = store.usage()["occupied_tokens"]

    child, child_logits = store.fork(parent.id, None)
    # 零拷贝：占用量不变，两个分支指向同一个物理 KvState，引用计数为 2。
    assert store.usage()["occupied_tokens"] == occupied_before
    snap = {b["id"]: b for b in store.debug_snapshot()["branches"]}
    assert snap[parent.id]["kv_id"] == snap[child.id]["kv_id"]
    assert snap[child.id]["kv_refcount"] == 2
    np.testing.assert_allclose(
        child_logits, forward_full(MODEL, [1, 2, 3, 4]), rtol=1e-5, atol=1e-6
    )
    _assert_invariants(store)


def test_fork_at_any_token_offset_and_recompute_consistency():
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([5, 6, 7, 8, 9])

    for offset in range(1, 6):
        child, logits = store.fork(parent.id, offset)
        assert child.token_ids == [5, 6, 7, 8, 9][:offset]
        np.testing.assert_allclose(
            logits,
            forward_full(MODEL, [5, 6, 7, 8, 9][:offset]),
            rtol=1e-5,
            atol=1e-6,
        )
        _assert_logits_match(store, child)
    _assert_invariants(store)


def test_append_on_shared_kv_only_stores_new_tokens():
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([1, 2, 3])
    child, _ = store.fork(parent.id, None)

    # 子分支追加：继承的前缀保持共享，只为新增 token 保存新 KV。
    _, logits = store.append(child.id, [4, 5])
    assert child.token_ids == [1, 2, 3, 4, 5]
    assert parent.token_ids == [1, 2, 3]
    _assert_logits_match(store, child)
    _assert_logits_match(store, parent)

    snap = {b["id"]: b for b in store.debug_snapshot()["branches"]}
    # 父分支的段仍与子分支共享；子分支新增 2 个位置。
    assert snap[parent.id]["kv_segments"][0]["kv_id"] == (
        snap[child.id]["kv_segments"][0]["kv_id"]
    )
    assert snap[parent.id]["kv_segments"][0]["refcount"] == 2
    assert store.usage()["occupied_tokens"] == 5
    _assert_invariants(store)


def test_truncated_fork_only_holds_visible_prefix():
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([1, 2, 3, 4, 5])
    child, _ = store.fork(parent.id, offset=2)  # 只继承 [1, 2]

    # 截断分叉不增加占用，且只登记前 2 行的视图；父分支仍保存 5 行。
    assert store.usage()["occupied_tokens"] == 5
    snap = {b["id"]: b for b in store.debug_snapshot()["branches"]}
    assert snap[child.id]["kv_segments"][0]["view_length"] == 2
    assert snap[parent.id]["kv_segments"][0]["segment_length"] == 5
    _assert_logits_match(store, child)

    # 子分支追加 1 个：共享前缀 5 行 + 新增 1 行 = 6。
    _, logits = store.append(child.id, [9])
    assert child.token_ids == [1, 2, 9]
    assert parent.token_ids == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(
        logits, forward_full(MODEL, [1, 2, 9]), rtol=1e-5, atol=1e-6
    )
    _assert_logits_match(store, child)
    _assert_logits_match(store, parent)
    assert store.usage()["occupied_tokens"] == 6
    _assert_invariants(store)


def test_parent_6_child_prefix_2_plus_1_then_close_parent():
    """需求示例：6 + 1 = 7；关闭父分支后只剩 2 + 1 = 3。"""
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([1, 2, 3, 4, 5, 6])
    child, _ = store.fork(parent.id, offset=2)
    assert store.usage()["occupied_tokens"] == 6

    store.append(child.id, [9])
    assert store.usage()["occupied_tokens"] == 7

    store.close(parent.id)
    # 父段尾部 4 行已无可见引用，立即裁剪；子段保留新增的 1 行。
    assert store.usage()["occupied_tokens"] == 3
    assert child.token_ids == [1, 2, 9]
    _assert_logits_match(store, child)
    _assert_invariants(store)


def test_close_releases_positions_when_last_visible_reference_gone():
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([1, 2, 3, 4, 5, 6])
    full, _ = store.fork(parent.id, None)
    trunc, _ = store.fork(parent.id, offset=3)

    # 关闭完整分叉：父分支与截断分叉都仍可见全部 6 行，不释放。
    store.close(full.id)
    assert store.usage()["occupied_tokens"] == 6

    # 关闭截断分叉：其可见的 3 行仍被父分支引用，不释放。
    store.close(trunc.id)
    assert store.usage()["occupied_tokens"] == 6

    # 关闭父分支：最后一组可见引用消失，立即归零。
    store.close(parent.id)
    assert store.usage()["occupied_tokens"] == 0
    assert store.usage()["kv_segment_count"] == 0
    _assert_invariants(store)


def test_close_releases_kv_after_last_reference():
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([1, 2, 3, 4])
    child, _ = store.fork(parent.id, None)

    # 关闭父分支：KV 仍被子分支引用，占用不释放。
    store.close(parent.id)
    assert store.usage()["occupied_tokens"] == 4
    assert child.kv is not None and child.kv.refcount == 1

    # 关闭最后一个引用：KV 释放，占用归零。
    store.close(child.id)
    assert store.usage()["occupied_tokens"] == 0
    assert store.usage()["branch_count"] == 0
    assert store.debug_snapshot()["occupied_matches_live_kv"]

def test_capacity_exceeded_rejects_whole_batch_atomically():
    store = BranchStore(MODEL, max_cached_tokens=10)
    branch, _ = store.create_branch([1, 2, 3])  # 占用 3
    tokens_before = list(branch.token_ids)
    occupied_before = store.usage()["occupied_tokens"]

    # 追加 8 个会到 11 > 10：整批拒绝。
    with pytest.raises(CapacityExceeded):
        store.append(branch.id, [4, 5, 6, 7, 8, 9, 10, 11])

    # 原分支与占用量保持原值。
    assert branch.token_ids == tokens_before
    assert store.usage()["occupied_tokens"] == occupied_before == 3
    _assert_invariants(store)

    # 恰好到上限的批次成功。
    store.append(branch.id, [4, 5, 6, 7, 8, 9, 10])
    assert store.usage()["occupied_tokens"] == 10
    _assert_logits_match(store, branch)

    # 再追加一个也被拒绝。
    with pytest.raises(CapacityExceeded):
        store.append(branch.id, [0])
    assert store.usage()["occupied_tokens"] == 10
    _assert_invariants(store)


def test_capacity_exceeded_with_shared_prefix_counts_only_new_tokens():
    """共享 KV 上追加：容量只看本批新增位置，不复制继承前缀。"""
    store = BranchStore(MODEL, max_cached_tokens=10)
    parent, _ = store.create_branch([1, 2, 3, 4, 5, 6])  # 占用 6
    child, _ = store.fork(parent.id, None)  # 共享，仍为 6

    # 只剩 4 个位置余量；追加 5 个 -> 11 > 10，整批拒绝。
    with pytest.raises(CapacityExceeded):
        store.append(child.id, [7, 8, 9, 10, 11])
    # 原子失败：仍是共享状态，占用不变。
    assert store.usage()["occupied_tokens"] == 6
    snap = {b["id"]: b for b in store.debug_snapshot()["branches"]}
    assert snap[parent.id]["kv_segments"][0]["kv_id"] == (
        snap[child.id]["kv_segments"][0]["kv_id"]
    )

    # 追加 4 个：仅新增 4 个位置，恰好 10，成功。
    store.append(child.id, [7, 8, 9, 10])
    assert store.usage()["occupied_tokens"] == 10
    assert child.token_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    _assert_logits_match(store, child)
    _assert_invariants(store)

    # 关闭父分支：子分支仍需要继承的 6 行前缀，占用不释放。
    store.close(parent.id)
    assert store.usage()["occupied_tokens"] == 10
    _assert_logits_match(store, child)
    _assert_invariants(store)

    # 关闭子分支：所有位置失去可见引用，占用才归零。
    store.close(child.id)
    assert store.usage()["occupied_tokens"] == 0
    _assert_invariants(store)


def test_create_context_over_capacity_rejected():
    store = BranchStore(MODEL, max_cached_tokens=4)
    with pytest.raises(CapacityExceeded):
        store.create_branch([1, 2, 3, 4, 5])
    assert store.usage()["occupied_tokens"] == 0
    assert store.usage()["branch_count"] == 0


def test_concurrent_appends_to_different_branches_stay_isolated():
    store = BranchStore(MODEL, max_cached_tokens=10_000)
    root, _ = store.create_branch([1, 2])
    children = [store.fork(root.id, None)[0] for _ in range(8)]
    store.close(root.id)  # 子分支各自持有引用

    errors: list[Exception] = []

    def worker(branch, token):
        try:
            for _ in range(20):
                store.append(branch.id, [token])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(b, i + 10))
        for i, b in enumerate(children)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    for i, b in enumerate(children):
        token = i + 10
        assert b.token_ids == [1, 2] + [token] * 20
        _assert_logits_match(store, b)
    _assert_invariants(store)


def test_concurrent_close_and_append_different_branches():
    """一个分支关闭、另一个分支追加，互不干扰。"""
    store = BranchStore(MODEL, max_cached_tokens=10_000)
    a, _ = store.create_branch([1, 2, 3])
    b, _ = store.create_branch([4, 5, 6])
    errors: list[Exception] = []

    def closer():
        try:
            store.close(a.id)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def appender():
        try:
            for _ in range(50):
                store.append(b.id, [7])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=closer)
    t2 = threading.Thread(target=appender)
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert errors == []
    assert b.token_ids == [4, 5, 6] + [7] * 50
    _assert_logits_match(store, b)
    assert store.usage()["occupied_tokens"] == 53
    _assert_invariants(store)


def test_concurrent_mixed_fork_append_close():
    """多线程随机混合 fork/append/close，全程不变量必须成立。"""
    import random

    store = BranchStore(MODEL, max_cached_tokens=100_000)
    root, _ = store.create_branch([1, 2])
    errors: list[Exception] = []

    def worker(seed):
        rng = random.Random(seed)
        parent = root.id
        for _ in range(30):
            try:
                op = rng.choice(["fork", "append", "close", "fork_append"])
                if op == "fork":
                    child, _ = store.fork(parent, None)
                    parent = child.id
                elif op == "fork_append":
                    child, _ = store.fork(parent, rng.choice([None, 1, 2]))
                    store.append(child.id, [rng.randrange(MODEL["vocab_size"])])
                    parent = child.id
                elif op == "append":
                    store.append(parent, [rng.randrange(MODEL["vocab_size"])])
                else:
                    store.close(parent)
                    child, _ = store.fork(root.id, None)
                    parent = child.id
            except KeyError:
                # 目标分支已被其他线程关闭：从根重新开始；根也没了就新建。
                try:
                    child, _ = store.fork(root.id, None)
                except KeyError:
                    child, _ = store.create_branch([1, 2])
                parent = child.id
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    _assert_invariants(store)
    # 存活分支的 logits 仍与整段重算一致（上下文未被其他线程污染）。
    for branch in list(store.debug_snapshot()["branches"]):
        live = store.get_branch(branch["id"])
        if live.token_ids:
            _assert_logits_match(store, live)


def test_unknown_branch_raises_keyerror():
    store = BranchStore(MODEL, max_cached_tokens=128)
    with pytest.raises(KeyError):
        store.append("nope", [1])
    with pytest.raises(KeyError):
        store.close("nope")
    with pytest.raises(KeyError):
        store.fork("nope", None)


def test_nested_truncated_forks_release_progressively():
    store = BranchStore(MODEL, max_cached_tokens=128)
    root, _ = store.create_branch([1, 2, 3, 4, 5, 6, 7, 8])
    mid, _ = store.fork(root.id, offset=6)
    leaf, _ = store.fork(mid.id, offset=3)
    assert store.usage()["occupied_tokens"] == 8  # 全部共享

    store.close(root.id)
    # 只剩 mid 可见前 6 行、leaf 可见前 3 行；尾部 2 行立即释放。
    assert store.usage()["occupied_tokens"] == 6
    _assert_invariants(store)

    store.close(mid.id)
    # 只剩 leaf 可见前 3 行，尾部 5 行立即释放。
    assert store.usage()["occupied_tokens"] == 3
    _assert_logits_match(store, leaf)
    _assert_invariants(store)

    store.close(leaf.id)
    assert store.usage()["occupied_tokens"] == 0
    _assert_invariants(store)


def test_interleaved_append_and_close_keep_usage_and_logits():
    """追加与关闭交错：用量、引用计数一致，每次追加的 logits 与整段重算一致。"""
    import random

    rng = random.Random(11)
    store = BranchStore(MODEL, max_cached_tokens=10_000_000)
    root, _ = store.create_branch([1, 2, 3, 4, 5, 6])

    live = [root]
    for step_index in range(200):
        action = rng.choice(["fork", "append", "close"])
        if action == "fork" and step_index % 7 != 0:
            parent = rng.choice(live)
            offset = rng.choice([None, 0, 1, len(parent.token_ids)])
            if offset is not None:
                offset = min(offset, len(parent.token_ids))
            child, logits = store.fork(parent.id, offset)
            if child.token_ids:
                np.testing.assert_allclose(
                    logits,
                    forward_full(MODEL, child.token_ids),
                    rtol=1e-5,
                    atol=1e-6,
                )
            live.append(child)
        elif action == "append":
            branch = rng.choice(live)
            batch = [rng.randrange(MODEL["vocab_size"]) for _ in range(3)]
            _, logits = store.append(branch.id, batch)
            np.testing.assert_allclose(
                logits,
                forward_full(MODEL, branch.token_ids),
                rtol=1e-5,
                atol=1e-6,
            )
        elif len(live) > 1:
            victim = rng.choice(live)
            live.remove(victim)
            store.close(victim.id)

        _assert_invariants(store)
        # 用量必须等于所有存活分支各自上下文长度去重前的“实际保存位置”：
        # 每份物理段只计一次，且等于该段的最大可见行数。
        snap = store.debug_snapshot()
        counted = sum(
            {
                seg["kv_id"]: seg["segment_length"]
                for b in snap["branches"]
                for seg in b["kv_segments"]
            }.values()
        )
        assert counted == snap["usage"]["occupied_tokens"]
        for b in live:
            if b.token_ids:
                _assert_logits_match(store, b)


def test_append_does_not_copy_inherited_prefix():
    """追加只写新段：继承段的数组对象与内容都保持不变。"""
    store = BranchStore(MODEL, max_cached_tokens=128)
    parent, _ = store.create_branch([1, 2, 3, 4])
    child, _ = store.fork(parent.id, None)

    inherited = child.parts[0][0]
    inherited_K = inherited.K
    inherited_V = inherited.V
    snapshot_K = inherited.K.copy()

    store.append(child.id, [5, 6])
    # 继承段没有被复制、改写或重新分配。
    assert child.parts[0][0] is inherited
    assert inherited.K is inherited_K
    assert inherited.V is inherited_V
    np.testing.assert_array_equal(inherited.K, snapshot_K)
    # 新段只包含本批 2 个 token。
    new_segment, new_view = child.parts[-1]
    assert new_segment is not inherited
    assert new_view == 2
    assert new_segment.length == 2
    _assert_logits_match(store, child)
    _assert_logits_match(store, parent)


def test_many_candidates_share_long_prefix_without_copying():
    """长上下文分叉出多条候选并各自续写：只增长新增 token 的位置数。"""
    store = BranchStore(MODEL, max_cached_tokens=1000)
    prefix = list(range(40))
    root, _ = store.create_branch(prefix)
    assert store.usage()["occupied_tokens"] == 40

    candidates = [store.fork(root.id, None)[0] for _ in range(5)]
    assert store.usage()["occupied_tokens"] == 40  # 分叉零拷贝

    for i, candidate in enumerate(candidates):
        _, logits = store.append(candidate.id, [i, i + 1])
        np.testing.assert_allclose(
            logits, forward_full(MODEL, prefix + [i, i + 1]), rtol=1e-5, atol=1e-6
        )
        # 每批只新增 2 个位置。
        assert store.usage()["occupied_tokens"] == 40 + 2 * (i + 1)
        _assert_invariants(store)

    # 关闭父分支：共享前缀仍被 5 条候选需要，只释放父分支的登记。
    store.close(root.id)
    assert store.usage()["occupied_tokens"] == 50
    for candidate in candidates:
        _assert_logits_match(store, candidate)
    _assert_invariants(store)

    # 逐个关闭候选：每个分支新增的 2 行立即释放；最后一个分支关闭时，
    # 共享的 40 行前缀也失去全部可见引用，一并释放。
    for i, candidate in enumerate(candidates):
        store.close(candidate.id)
        expected = 0 if i == len(candidates) - 1 else 50 - 2 * (i + 1)
        assert store.usage()["occupied_tokens"] == expected
        _assert_invariants(store)
    assert store.usage()["occupied_tokens"] == 0
