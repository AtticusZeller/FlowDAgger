# Bug Journal

> 开发过程中的硬核经验：触发情况、解决方案、原因解释。

<!-- 新 bug 追加到本行下方 -->

## 2026-07-30 · 录像复用策略预处理帧导致低清俯视证据

- 触发：评估把 `obs_to_img()` 的 128×128 steering 输入直接写入视频，并沿用
  `corner3` 策略机位。
- 现象：视频分辨率低且偏俯视，无法清楚判断机械臂与目标物的交互。
- 处理：MetaWorld adapter 新增共享同一 model/data 的独立 MuJoCo renderer；
  策略输入保持不变，视频单独使用 `corner`、640×480、30 FPS。只为配置要求的
  episode 执行高清渲染，避免改变 25-episode 评估统计口径。
- 原因：模型输入证据与人类视觉证据用途不同，不应共享经过网络预处理的帧。

## 2026-07-21 · JAX BC update 的 contracting dimension 错误

- 触发：当前 DSW 环境安装 JAX `0.8.0` 后，FlowDAgger short 配置首次 BC update 使用 `bc_batch_size=64`。
- 现象：`steering_policy.py:195` 抛出 `jax.errors.JaxRuntimeError: CANCELLED: Too small divisible part of the contracting dimension.`，进程返回码为 `1`。
- 已验证缓解：保持其他参数不变，将 batch size 调为 `16` 后，100 BC steps、两次 5-episode eval 和 checkpoint100 均通过。
- 未决原因：尚未确认是 JAX/XLA、模型维度还是当前依赖版本组合导致；默认 `256` 仍不可假设可用，后续 full run 需显式选择 batch size 并记录。
