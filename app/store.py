"""分支与共享 KV 缓存存储。

核心模型
--------
- 物理 KV 按“段”（:class:`KvSegment`）存放，每段是一段连续的 K/V 数组。
  一个段可以被多个分支同时引用，每个分支只看到该段的一个**前缀视图**
  （``segment.views[branch_id]`` 记录该分支可见的行数）。
- 分支（:class:`Branch`）的上下文是一串有序的 ``(段, 可见长度)``。分叉只复制
  这串视图描述，不复制任何 K/V 数据：完整分叉看到父分支的全部段，截断分叉
  只保留可见前缀，父分支中更靠后的段完全不被新分支引用。
- 追加 token 时，继承来的段保持原样共享，只有本批新增 token 的 K/V 写入一个
  新段。因此一次追加最多新增 ``len(batch)`` 个缓存位置，不会复制前缀。
- 每个段只按“物理保存的行数”计入一次用量；共享不重复计数。
- 某个位置失去最后一个可见引用时立即回收：段的引用全部消失就释放整段；
  仍有分支引用但可见前缀变短时，把段裁剪到最长可见前缀并立即回收多余行。
  例如父分支 6 个 token、子分支继承前 2 个再追加 1 个：占用 6 + 1 = 7；
  关闭父分支后，父段裁到子分支可见的 2 行，加上子分支新增的 1 行，共 3。
- 容量超限在任何状态修改之前检查，整批拒绝（HTTP 413），分支内容、KV 引用
  和全局用量保持原值。
- logits 直接在各段可见前缀上计算，不把历史拼接成连续数组。

线程模型：单进程服务，所有公开方法在同一把可重入锁内完成“检查 + 修改”，
保证每个操作原子可见。每步仅为小规模 NumPy 运算，串行化的开销可以忽略。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

import numpy as np

from . import model as model_mod


class CapacityExceeded(Exception):
    """追加整批 token 后会超出缓存上限。"""


@dataclass
class KvSegment:
    """一段物理 K/V 存储，可被多个分支按前缀视图共享。

    ``length`` 是物理保存的行数；``views`` 是 ``分支 ID -> 可见行数``。
    不变量：``length == max(views.values())``（没有引用时 ``length == 0``），
    因此物理存储中不会残留没有任何分支可见的尾部行。
    """

    K: np.ndarray | None
    V: np.ndarray | None
    length: int
    views: dict[str, int] = field(default_factory=dict)

    @property
    def refcount(self) -> int:
        return len(self.views)

    @property
    def max_view(self) -> int:
        return max(self.views.values()) if self.views else 0


# 旧名称的兼容别名（物理 KV 现在按段管理）。
KvState = KvSegment


@dataclass
class Branch:
    id: str
    token_ids: list[int]
    # 有序的 (物理段, 本分支可见的前缀行数)
    parts: list[tuple[KvSegment, int]] = field(default_factory=list)

    @property
    def kv(self) -> KvSegment | None:
        """最近一段物理 KV；仅用于调试，不代表整个分支上下文。"""
        return self.parts[-1][0] if self.parts else None


class BranchStore:
    def __init__(self, model: dict, max_cached_tokens: int):
        self._model = model
        self._max_cached_tokens = int(max_cached_tokens)
        self._branches: dict[str, Branch] = {}
        self._occupied = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 元数据

    @property
    def max_cached_tokens(self) -> int:
        return self._max_cached_tokens

    def usage(self) -> dict:
        with self._lock:
            live = self._live_segments()
            return {
                "occupied_tokens": self._occupied,
                "max_cached_tokens": self._max_cached_tokens,
                "branch_count": len(self._branches),
                "kv_state_count": len(live),
                "kv_segment_count": len(live),
            }

    def _live_segments(self) -> list[KvSegment]:
        """所有仍被至少一个分支引用的物理段（按对象去重）。"""
        seen: dict[int, KvSegment] = {}
        for branch in self._branches.values():
            for segment, _view in branch.parts:
                seen[id(segment)] = segment
        return list(seen.values())

    def _get_branch(self, branch_id: str) -> Branch:
        branch = self._branches.get(branch_id)
        if branch is None:
            raise KeyError(branch_id)
        return branch

    def get_branch(self, branch_id: str) -> Branch:
        """按 ID 取分支，不存在则抛 KeyError。"""
        with self._lock:
            return self._get_branch(branch_id)

    # ------------------------------------------------------------- 引用计数管理

    def _new_segment(self, K: np.ndarray, V: np.ndarray) -> KvSegment:
        segment = KvSegment(K=K, V=V, length=int(K.shape[0]))
        self._occupied += segment.length
        return segment

    def _retain_view(
        self, segment: KvSegment, branch_id: str, view: int
    ) -> tuple[KvSegment, int]:
        """登记一个分支对某段前 ``view`` 行的可见引用（不增加物理占用）。"""
        assert 0 < view <= segment.length
        assert branch_id not in segment.views  # 每个分支对每段只登记一次
        segment.views[branch_id] = view
        return segment, view

    def _release_view(self, segment: KvSegment, branch_id: str) -> None:
        """撤销一个可见引用；位置失去最后一个可见引用时立即回收。"""
        view = segment.views.pop(branch_id, None)
        if view is None:  # 同一分支对同一段只登记一次，防御性处理
            return
        if not segment.views:
            # 最后一个可见引用消失：整段释放，占用立即归零。
            self._occupied -= segment.length
            segment.K = None
            segment.V = None
            segment.length = 0
            return

        max_view = segment.max_view
        if max_view < segment.length:
            # 尾部行已无任何分支可见：裁剪并复制前缀，释放旧数组。
            self._occupied -= segment.length - max_view
            segment.K = np.array(segment.K[:max_view], copy=True)
            segment.V = np.array(segment.V[:max_view], copy=True)
            segment.length = max_view

    # ----------------------------------------------------------- 上下文与 logits

    @staticmethod
    def _collect_prefix(
        parts: list[tuple[KvSegment, int]], length: int
    ) -> list[tuple[KvSegment, int]]:
        """从 ``parts`` 中截取前 ``length`` 个 token 对应的有序视图。"""
        selected: list[tuple[KvSegment, int]] = []
        remaining = length
        for segment, view in parts:
            if remaining <= 0:
                break
            take = min(view, remaining)
            selected.append((segment, take))
            remaining -= take
        if remaining != 0:
            raise AssertionError("prefix length exceeds available KV views")
        return selected

    @staticmethod
    def _visible_parts(parts: list[tuple[KvSegment, int]]):
        """按上下文顺序给出各段可见前缀的 ``(K, V)``，供 logits 计算使用。"""
        return [(segment.K[:view], segment.V[:view]) for segment, view in parts]

    def _logits_at(self, branch: Branch) -> np.ndarray | None:
        """回放分支最后一个 token 的 next-token logits。

        只使用分支可见的继承前缀与自身新增的 KV，结果与整段重算一致；
        不复制、不拼接历史数组。
        """
        if not branch.token_ids:
            return None
        visible = self._visible_parts(branch.parts)
        total = sum(K.shape[0] for K, _V in visible)
        if total != len(branch.token_ids):
            raise AssertionError("visible KV length does not match branch context")
        return model_mod.logits_over_parts(
            self._model, branch.token_ids[-1], visible
        )

    # --------------------------------------------------------------- 分支生命周期

    def create_branch(
        self, token_ids: list[int]
    ) -> tuple[Branch, np.ndarray | None]:
        """提交一批 token 建立上下文，返回新分支与最后位置的 next-token logits。"""
        with self._lock:
            batch = len(token_ids)
            # 容量预检先于任何状态修改；新上下文只保存 batch 个位置。
            if self._occupied + batch > self._max_cached_tokens:
                raise CapacityExceeded(
                    f"creating context with {batch} tokens would raise occupancy "
                    f"{self._occupied} -> {self._occupied + batch}, "
                    f"limit {self._max_cached_tokens}"
                )

            branch = Branch(id=uuid.uuid4().hex, token_ids=list(token_ids))
            logits = None
            if batch:
                K, V = model_mod.encode_batch(self._model, token_ids)
                segment = self._new_segment(K, V)
                branch.parts.append(self._retain_view(segment, branch.id, batch))
                logits = model_mod.logits_from_kv(
                    self._model, token_ids[-1], segment.K, segment.V
                )
            self._branches[branch.id] = branch
            return branch, logits

    def fork(
        self, parent_id: str, offset: int | None
    ) -> tuple[Branch, np.ndarray | None]:
        """从父分支的任一已有 token 处分叉，零拷贝继承 KV 视图。

        offset 为保留的 token 数：新分支上下文 = 父分支前 offset 个 token。
        省略 offset 时继承完整上下文。截断分叉只引用可见前缀所在的段，
        不增加任何缓存占用，也不引用父分支中不可见的部分。
        """
        with self._lock:
            parent = self._get_branch(parent_id)
            if offset is None:
                offset = len(parent.token_ids)
            if offset < 0 or offset > len(parent.token_ids):
                raise ValueError(
                    f"offset {offset} out of range [0, {len(parent.token_ids)}]"
                )

            child = Branch(
                id=uuid.uuid4().hex, token_ids=list(parent.token_ids[:offset])
            )
            # 只挑选可见前缀覆盖的段；父分支更靠后的段不被引用。
            selected = self._collect_prefix(parent.parts, offset)
            for segment, view in selected:
                self._retain_view(segment, child.id, view)
            child.parts = selected
            logits = self._logits_at(child)
            self._branches[child.id] = child
            return child, logits

    def append(
        self, branch_id: str, token_ids: list[int]
    ) -> tuple[Branch, np.ndarray]:
        """向分支批量追加 token；超限时整批拒绝，状态保持不变。

        继承前缀保持共享；本批 token 的 K/V 写入一个新段，只新增
        ``len(token_ids)`` 个缓存位置。
        """
        with self._lock:
            if not token_ids:
                raise ValueError("token batch must not be empty")

            branch = self._get_branch(branch_id)
            batch = len(token_ids)

            # ---- 容量预检（在任何状态修改之前）----
            # 追加不会复制继承前缀，占用只按本批新增位置增长。
            projected = self._occupied + batch
            if projected > self._max_cached_tokens:
                raise CapacityExceeded(
                    f"appending {batch} tokens would raise occupancy "
                    f"{self._occupied} -> {projected}, "
                    f"limit {self._max_cached_tokens}"
                )

            # ---- 预检通过，执行追加 ----
            # 新 token 的 K/V 与历史无关，直接计算并写入新段。
            K_new, V_new = model_mod.encode_batch(self._model, token_ids)
            visible = self._visible_parts(branch.parts)
            visible.append((K_new, V_new))
            logits = model_mod.logits_over_parts(
                self._model, token_ids[-1], visible
            )

            segment = self._new_segment(K_new, V_new)
            branch.parts.append(self._retain_view(segment, branch.id, batch))
            branch.token_ids.extend(token_ids)
            return branch, logits

    def close(self, branch_id: str) -> None:
        """关闭分支；失去最后一个可见引用的 KV 位置立即释放。"""
        with self._lock:
            branch = self._branches.pop(branch_id, None)
            if branch is None:
                raise KeyError(branch_id)
            for segment, _view in branch.parts:
                self._release_view(segment, branch.id)
            branch.parts.clear()

    # ------------------------------------------------------------------- 调试

    def debug_snapshot(self) -> dict:
        """返回内部状态快照，供测试与排障核对引用计数/占用量不变量。"""
        with self._lock:
            live = self._live_segments()
            counted = sum(segment.length for segment in live)
            stored_rows = sum(
                segment.K.shape[0] if segment.K is not None else 0
                for segment in live
            )
            return {
                "usage": self.usage(),
                "occupied_matches_live_kv": counted == self._occupied,
                # 实际分配的行数也必须等于用量：不存在无人可见的残留行。
                "occupied_matches_stored_rows": stored_rows == self._occupied,
                "branches": [
                    {
                        "id": branch.id,
                        "tokens": list(branch.token_ids),
                        "view_length": len(branch.token_ids),
                        # 该分支引用的各段物理行数之和；截断视图可能小于段长度。
                        "kv_length": sum(
                            segment.length for segment, _view in branch.parts
                        ),
                        "kv_id": (
                            id(branch.parts[-1][0]) if branch.parts else None
                        ),
                        "kv_refcount": (
                            branch.parts[-1][0].refcount if branch.parts else 0
                        ),
                        "kv_segments": [
                            {
                                "kv_id": id(segment),
                                "segment_length": segment.length,
                                "view_length": view,
                                "refcount": segment.refcount,
                            }
                            for segment, view in branch.parts
                        ],
                    }
                    for branch in self._branches.values()
                ],
            }
