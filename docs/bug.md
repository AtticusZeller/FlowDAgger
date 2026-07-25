# Bug Journal

> 开发过程中的硬核经验：触发情况、解决方案、原因解释。

<!-- 新 bug 追加到本行下方 -->

## 2026-07-21 · JAX BC update 的 contracting dimension 错误

- 触发：当前 DSW 环境安装 JAX `0.8.0` 后，FlowDAgger short 配置首次 BC update 使用 `bc_batch_size=64`。
- 现象：`steering_policy.py:195` 抛出 `jax.errors.JaxRuntimeError: CANCELLED: Too small divisible part of the contracting dimension.`，进程返回码为 `1`。
- 已验证缓解：保持其他参数不变，将 batch size 调为 `16` 后，100 BC steps、两次 5-episode eval 和 checkpoint100 均通过。
- 未决原因：尚未确认是 JAX/XLA、模型维度还是当前依赖版本组合导致；默认 `256` 仍不可假设可用，后续 full run 需显式选择 batch size 并记录。
