# shared 模块

该目录提供不同后端共用的 MetaWorld 任务和专家组件：

- `task_configs.py`：任务注册表；当前注册论文 MetaWorld-12 任务，统一使用
  `pi05_metaworld` 与 MT50 prompt，并分别声明策略相机和视频相机。
- `experts/`：读取模拟器特权状态并输出纠正动作的 scripted expert。
- `intervention_handler.py`：根据 beta decay 或 action disagreement 决定专家接管，并记录反演噪声目标。

模块被 `flowdagger_pi05` 训练入口直接导入；新增任务时需要同时确认环境 ID、prompt、
scripted policy、相机配置和 checkpoint 配置，并运行 expert probe。
