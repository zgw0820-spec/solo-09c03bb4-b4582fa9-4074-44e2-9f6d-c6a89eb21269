# NumPy 单头因果注意力 · 分支化解码 API

单进程 FastAPI 解码服务，内置固定权重（纯 NumPy 实现）的单层因果注意力模型，
支持从同一上下文分叉出多条候选、批量追加 token、关闭分支，并管理共享的
K/V 缓存。

## 模型

- 结构：token embedding → 单头因果自注意力（softmax，缩放点积）→ 残差 + 输出投影 → logits
- 权重由固定种子 `MODEL_SEED` 生成（默认 `20261120`），进程内恒定
- 默认词表 64、隐藏维度 16（可用环境变量调整，见下）
- 提供两条前向路径，next-token logits 数值一致（测试容差 rtol=1e-5）：
  - 增量路径：`model.step`，复用历史 K/V，每步只计算新 token；分支存储用
    `model.encode_batch` 只为新 token 生成 K/V，用 `model.logits_over_parts`
    跨多段 KV 直接计算 logits，不拼接、不复制继承前缀
  - 基准路径：`model.forward_full`，对整段 token 一次性重算并施加因果掩码

## 分支与缓存语义

- **建立上下文**：`POST /branches` 提交 token ID 序列（允许空数组）
- **分叉**：`POST /branches/{id}/fork`，`offset` 可指向任一已有 token 位置
  （省略则继承完整上下文）。分叉零拷贝复用继承的物理 KV 视图，仅登记
  引用，不增加缓存占用
- **批量追加**：`POST /branches/{id}/tokens`
  - 继承来的 KV 保持共享，**只为本批新增 token 保存新的 K/V 段**，
    一次追加最多新增 `len(batch)` 个缓存位置，不复制前缀
  - 截断分叉只持有可见的前缀视图；父分支中更靠后的位置完全不被引用
- **关闭**：`DELETE /branches/{id}`；某个位置失去最后一个可见引用时立即
  释放对应存储：整段无人可见就整段释放，仅尾部无人可见就把段裁剪到最长
  可见前缀并回收多余行
- **容量**：按实际保存的 token 位置数计（所有存活物理 KV 段的行数之和，
  共享不重复计数，截断不占额外空间）。超限的提交在任何状态修改之前检查，
  **整批拒绝**（HTTP 413），原分支内容、KV 引用与全局占用量保持不变
- **并发**：单进程内所有操作为同一把可重入锁保护的原子操作；
  追加与关闭交错执行时，缓存用量与引用计数仍保持一致
- 每次成功响应都返回当前缓存用量；建立上下文 / 分叉 / 追加还返回
  `logits`（增量）与 `full_recompute_logits`（整段重算），两者一致

### 需求示例（也是端到端脚本场景 A）

| 步骤 | 操作 | 占用位置数 |
| --- | --- | --- |
| 1 | 父分支提交 6 个 token | 6 |
| 2 | 从 offset=2 分叉（只持有前 2 个的视图） | 6 |
| 3 | 子分支追加 1 个 token（只保存这 1 个的新 KV） | **7** |
| 4 | 关闭父分支（其不可见的尾部 4 行立即回收） | **3** |

## 用 Docker Compose 启动

```bash
docker compose up --build
```

- 服务：http://localhost:8000
- Swagger UI：http://localhost:8000/docs （`/` 自动重定向到 `/docs`）
- OpenAPI：http://localhost:8000/openapi.json
- 容器固定单 uvicorn worker，保证进程内缓存与锁状态一致

### 可配置环境变量（docker-compose.yml）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MAX_CACHED_TOKENS` | `128` | 缓存上限（实际保存的 token 位置数） |
| `MODEL_VOCAB_SIZE` | `64` | 词表大小 |
| `MODEL_D_MODEL` | `16` | 模型隐藏维度 |
| `MODEL_SEED` | `20261120` | 固定权重随机种子 |

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/branches` | 建立上下文，body `{"token_ids": [...]}` |
| POST | `/branches/{id}/fork` | 分叉，body `{"offset": 3}` 或 `{}` |
| POST | `/branches/{id}/tokens` | 批量追加，body `{"token_ids": [...]}` |
| DELETE | `/branches/{id}` | 关闭分支 |
| GET | `/stats` | 缓存用量 |
| GET | `/branches` | 调试快照（KV 段、每个分支的可见长度、引用计数、占用不变量） |
| GET | `/health` | 健康检查 |

在 Swagger 中依次调用示例（分叉、分别续写、关闭后释放缓存）：

1. `POST /branches`，body `{"token_ids": [1, 2, 3, 4, 5, 6]}` → 记下 `branch_id`，
   `usage.occupied_tokens` 为 6
2. `POST /branches/{id}/fork`，body `{}` → 完整分叉，占用仍为 6，
   `/branches` 中可见两个分支共享同一个 `kv_id`
3. `POST /branches/{id}/fork`，body `{"offset": 2}` → 截断分叉，占用仍为 6，
   快照里该分支的 `view_length` 为 2
4. 对上一步的子分支 `POST /branches/{id}/tokens`，body `{"token_ids": [7, 8]}`
   → 只新增 2 个位置，占用 8；返回的 `logits` 与 `full_recompute_logits` 一致
5. `DELETE` 关闭完整分叉与父分支 → 前缀段裁剪到截断分支仍可见的 2 行，
   加上子分支新增的 2 行，占用降为 4
6. 关闭剩余分支 → `/stats` 的 `occupied_tokens` 回到 0

## 本地开发与测试（无 Docker 时）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest            # 30 个单元/API 测试
MAX_CACHED_TOKENS=20 .venv/bin/python -m uvicorn app.main:app --port 8000
.venv/bin/python tools/e2e_check.py   # 对运行中的服务做真实 HTTP 端到端核对
```

`requirements.txt` 为镜像运行依赖；`requirements-dev.txt` 额外包含 pytest 与
httpx（Starlette TestClient 所需）。

## 目录结构

```
app/
  model.py   # 固定权重模型、增量 step、分段 logits、整段重算 forward_full
  store.py   # 分支/KV 段、可见前缀视图、引用回收、容量控制、并发锁
  main.py    # FastAPI 端点、请求校验、错误响应
tests/       # pytest：模型一致性 / 存储语义 / 并发 / HTTP API / 413 原子性
tools/       # 端到端核对脚本（非镜像内容）
Dockerfile, docker-compose.yml, requirements.txt
```

## 错误响应

- `413 capacity_exceeded`：批次超限被整批拒绝，响应含当前 `usage`
- `404 branch_not_found`：分支不存在（可能已被关闭）
- `422`：请求体校验失败（空追加批次、token 越界、offset 越界）
