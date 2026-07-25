# Command Reference

> 项目常用命令与用户侧验证入口。命令应可直接复制执行。

## 常用命令

### 当前可复现实验

当前代码库没有 `dev.sh`、YAML 实验配置或 sweep 脚本。可由代码确认的官方组合只有：

- 环境：`metaworld`。
- 任务：`metaworld_assembly`，对应 MetaWorld V3 `assembly-v3`。
- 基础策略：`pi05`，对应 openpi 配置 `pi05_metaworld`。
- 默认算法路径：启用 scripted expert intervention，`beta_decay`，`perstep_fp` 噪声反演。
- 权重：首次运行自动从 Hugging Face 下载 `mmurray-ms/pi05-metaworld`；也可以用 `METAWORLD_CHECKPOINT` 或 `--openpi_checkpoint` 指向本地 checkpoint。

代码还暴露了以下可运行的 ablation 轴，但它们不是仓库提供的官方实验 sweep：

- `--use_interventions 0`：不启用专家接管，可作为 base-policy/无 intervention 对照。
- `--intervention_mode beta_decay|disagreement`：两种接管触发方式。
- `--inversion_method none|euler_reverse|adam|hybrid|fixed_point|perstep_fp`：六种反演路径；默认是 `perstep_fp`。
- `--eval_only_ckpt <dir>`：只恢复 steering checkpoint 并执行评估。

### W&B 训练记录

当前在线记录配置为：

- Entity：`1831768457`
- Project：`flowdagger`
- 训练命令必须设置 `WANDB_MODE=online`，并传入 `--wandb_entity 1831768457 --wandb_project flowdagger`。
- 必须提供非空 `--prefix`；训练脚本以它作为 W&B run 名称前缀。
- 前面三个实验已从本地 offline run 同步到 W&B；后续实验直接在线记录。

登录凭证从本机 W&B 标准认证配置读取，不把 API key 写入命令、代码或文档。远端 run URL 会写入对应的实验结果 Markdown。

### 依赖安装（Python 3.11）

本机已有 `/root/miniconda3/envs/dsrl_pi0`，其 Python 版本为 3.11，优先复用它，不在 Python 3.13 的 base 环境中安装。依赖安装顺序必须保持：先初始化并安装 openpi，再安装本项目 requirements，使本项目 pin 生效。

```bash
cd /root/FlowDAgger
findmnt -T /mnt/data
git submodule update --init flowdagger_pi05/openpi
# DSW 默认镜像可能缺少 hatchling/setuptools；使用官方 PyPI 完成 build isolation。
/root/miniconda3/envs/dsrl_pi0/bin/python -m pip install --index-url https://pypi.org/simple hatchling setuptools
/root/miniconda3/envs/dsrl_pi0/bin/python -m pip install --index-url https://pypi.org/simple -e flowdagger_pi05/openpi
/root/miniconda3/envs/dsrl_pi0/bin/python -m pip install --index-url https://pypi.org/simple -r flowdagger_pi05/requirements.txt
```

依赖安装完成后的导入级验证：

```bash
cd /root/FlowDAgger
/root/miniconda3/envs/dsrl_pi0/bin/python -m pip check
/root/miniconda3/envs/dsrl_pi0/bin/python flowdagger_pi05/train_flowdagger.py --help
/root/miniconda3/envs/dsrl_pi0/bin/python - <<'PY'
import jax
import metaworld
import mujoco
import openpi

print("JAX devices:", jax.devices())
print("MetaWorld assembly:", "assembly-v3" in metaworld.ALL_V3_ENVIRONMENTS)
print("MuJoCo:", mujoco.__version__)
print("openpi:", openpi.__file__)
PY
```

### 实验输出目录与后台执行

重要实验输出统一放在可持久化 OSS 挂载下；运行前必须确认 `findmnt` 显示 `/mnt/data` 为可写：

```bash
findmnt -T /mnt/data
mkdir -p /mnt/data/atticux/FlowDAgger/{runs,results,checkpoints}
```

训练脚本使用 `EXP` 选择输出根目录。下面的 `tmux` 命令会创建后台会话；每次实验使用独立的 `RUN_ID`、日志文件和结果目录：

```bash
RUN_ID=exp-01-smoke
RUN_DIR=/mnt/data/atticux/FlowDAgger/runs/$RUN_ID
WANDB_ENTITY=1831768457
WANDB_PROJECT=flowdagger
mkdir -p "$RUN_DIR"
tmux new-session -d -s "$RUN_ID" \
  "cd /root/FlowDAgger/flowdagger_pi05 && \
   MUJOCO_GL=egl EXP=$RUN_DIR WANDB_MODE=online \
   /root/miniconda3/envs/dsrl_pi0/bin/python train_flowdagger.py \
     --env metaworld --task_key metaworld_assembly --seed 42 \
     --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT --prefix $RUN_ID \
     2>&1 | tee $RUN_DIR/console.log"
tmux ls
tmux attach -t "$RUN_ID"                    # 需要查看时使用 Ctrl-b d 脱离
tail -f "$RUN_DIR/console.log"               # 不进入会话也可查看
```

每次实验结束后，单独创建 `/mnt/data/atticux/FlowDAgger/results/<RUN_ID>.md`，记录命令、提交版本、依赖版本、开始/结束时间、返回码、成功率、反演误差和失败原因。`console.log` 只作为原始证据，不替代结果摘要。

### 实验 0：最小 FlowDAgger smoke

目标是验证 checkpoint 下载、MetaWorld 环境、pi0.5 推理、专家接管、一次 per-step inversion、一次 BC update 和一次单 episode eval。它不是论文统计结果。

```bash
RUN_ID=exp-01-smoke
RUN_DIR=/mnt/data/atticux/FlowDAgger/runs/$RUN_ID
mkdir -p "$RUN_DIR"
tmux new-session -d -s "$RUN_ID" \
  "cd /root/FlowDAgger/flowdagger_pi05 && \
   MUJOCO_GL=egl EXP=$RUN_DIR WANDB_MODE=online \
   /root/miniconda3/envs/dsrl_pi0/bin/python train_flowdagger.py \
     --env metaworld --task_key metaworld_assembly --seed 42 \
     --wandb_entity 1831768457 --wandb_project flowdagger \
     --max_steps 0 --seed_expert_episodes 1 \
     --bc_steps_per_episode 1 --bc_batch_size 16 \
     --eval_episodes 1 --eval_interval 1 --log_interval 1 \
     --inversion_method perstep_fp --fp_per_step 1 \
     --save_eval_video 0 --render 0 --prefix exp-01-smoke \
     2>&1 | tee $RUN_DIR/console.log"
```

静态估计：冷启动主要耗时在 checkpoint 下载和 JAX 编译，通常按 5–30 分钟量级观察；缓存与编译缓存命中后，单次 smoke 预计数分钟。实际时间以 `console.log` 为准。

### 实验 1：短程 sanity run

仅在实验 0 完成且输出中没有导入、GPU、环境或 inversion 错误时运行。它用约 100 个 BC steps 检查 loss、intervention、eval 和 checkpoint 是否持续工作：

```bash
RUN_ID=exp-02-short-b16
RUN_DIR=/mnt/data/atticux/FlowDAgger/runs/$RUN_ID
mkdir -p "$RUN_DIR"
tmux new-session -d -s "$RUN_ID" \
  "cd /root/FlowDAgger/flowdagger_pi05 && \
   MUJOCO_GL=egl EXP=$RUN_DIR WANDB_MODE=online \
   /root/miniconda3/envs/dsrl_pi0/bin/python train_flowdagger.py \
     --env metaworld --task_key metaworld_assembly --seed 42 \
     --wandb_entity 1831768457 --wandb_project flowdagger \
     --max_steps 100 --seed_expert_episodes 1 \
     --bc_steps_per_episode 10 --bc_batch_size 16 \
     --eval_episodes 5 --eval_interval 100 --log_interval 10 \
     --checkpoint_interval 100 --save_eval_video 0 --render 0 \
     --inversion_method perstep_fp --prefix exp-02-short-b16 \
     2>&1 | tee $RUN_DIR/console.log"
```

实测（缓存已命中）：约 5 分 12 秒；包含 10 个在线 episode、step 0/100 各 5 个 eval episode，并成功写出 checkpoint100。首次冷启动还需额外考虑 checkpoint 下载和 JAX 编译时间。

### 实验 2：官方默认全量

只有短程 sanity run 通过且 GPU/OSS 资源稳定时再启动。下面保留代码默认值，使用 `seed=42`，并在线记录到 W&B。注意：代码默认 `bc_batch_size=256`；当前环境中 `bc_batch_size=64` 已在首次 BC update 触发 JAX `Too small divisible part of the contracting dimension`，因此这条命令目前属于“官方配置、未验证”，不建议直接后台启动：

```bash
RUN_ID=exp-03-default
RUN_DIR=/mnt/data/atticux/FlowDAgger/runs/$RUN_ID
mkdir -p "$RUN_DIR"
tmux new-session -d -s "$RUN_ID" \
  "cd /root/FlowDAgger/flowdagger_pi05 && \
   MUJOCO_GL=egl EXP=$RUN_DIR WANDB_MODE=online \
   /root/miniconda3/envs/dsrl_pi0/bin/python train_flowdagger.py \
     --env metaworld --task_key metaworld_assembly --seed 42 \
     --wandb_entity 1831768457 --wandb_project flowdagger \
     --prefix exp-03-default --save_eval_video 0 --render 0 \
     2>&1 | tee $RUN_DIR/console.log"
```

默认规模约为 4000 BC steps、40 个训练 episode、10 个 seed-expert episode，并在 step 0 和每 500 steps 做 25 episode eval（共约 225 个 eval episode）；根据 100-step 短程实测，默认全量至少是小时级任务。建议先将命令中的 `--bc_batch_size 256` 覆盖为已验证的 `--bc_batch_size 16`，并确认接受这不是严格的原始默认配置后再启动。

若确认采用已验证的稳定化配置，只需在上面命令的训练参数中增加：

```text
--bc_batch_size 16 --checkpoint_interval 500
```

并将 `RUN_ID`、`RUN_DIR` 和 `--prefix` 改为 `exp-03-default-b16`，避免与官方默认配置混淆。

本次实际启动实例：`exp-03-default-b16`，W&B run 为 <https://wandb.ai/1831768457/flowdagger/runs/exp-03-default-b16_2026_07_21_19_37_04_0000--s-42>，本地日志位于 `/mnt/data/atticux/FlowDAgger/runs/exp-03-default-b16/console.log`。

### 后台任务检查与停止

```bash
tmux ls
tmux capture-pane -pt exp-01-smoke -S -80
test -f /mnt/data/atticux/FlowDAgger/runs/exp-01-smoke/console.log && \
  tail -n 80 /mnt/data/atticux/FlowDAgger/runs/exp-01-smoke/console.log
tmux kill-session -t exp-01-smoke       # 仅在确认需要停止时执行
```

## 待用户验证

- **状态**：依赖、GPU 导入、smoke、short 和 `bc_batch_size=16` 的 full 均已通过；严格官方默认 `bc_batch_size=256` 仍未验证。
- **目的**：如需严格官方默认复现，再单独确认 batch size 256 的兼容性；当前推荐复用已验证的 batch size 16 配置。
- **依赖**：Python 3.11 环境、CUDA/JAX GPU、MetaWorld/MuJoCo、可访问 Hugging Face checkpoint，以及当前 W&B entity/project 的在线权限。
- **命令**：稳定化 full 命令已执行完成；后续复现实验复制“实验 2”命令，增加 `--bc_batch_size 16 --checkpoint_interval 500`，并使用新的运行 ID；运行前确认 `findmnt -T /mnt/data` 为可写。
- **通过标准**：命令返回码为 0；日志包含 pi0.5 checkpoint 加载、MetaWorld 任务初始化、无未捕获异常；结果文件记录 eval success rate、inversion 指标和 checkpoint。
- **失败返回**：对应 `RUN_ID` 的 `console.log`、结果 Markdown、`pip check` 输出、`jax.devices()` 输出和最后 100 行日志。
