# 阶段 2C GitHub 同步报告

核查日期 2026-11-23。阶段 2C 代码 commit 已推送到 `origin/feature/clip-enhancement-v1-ssh-pilot`，并通过 `git ls-remote` 核对。云端运行应固定到下表的代码 commit。 [GitHub 分支](https://github.com/bityang15/CLIP-Cloud-Experiment/tree/feature/clip-enhancement-v1-ssh-pilot) · [代码 commit](https://github.com/bityang15/CLIP-Cloud-Experiment/commit/23261793e4cfcab95ab82d4c23687c255c7d3c5a)

| 项目 | 已确认值 |
| --- | --- |
| 可执行代码 commit | `23261793e4cfcab95ab82d4c23687c255c7d3c5a` |
| 基线与远端 `main` | `b490095904883ffe36bfb755a78b9dff2c485a8a` |
| 增强协议 JSON SHA-256 | `3d869d46ab0b75a6a1e5535227685edd4a1c8e9cc18912428363c71edadca8fe` |
| 预检输入锁 JSON SHA-256 | `20a5febf8de8bed92fe953b4642150ec4fcce4596ad8796e2be535c59d45bc7c` |

代码提交暂存差异共 13 个文件，含增强协议、两套 Unsharp 参数评审、独立入口、测试、参考哈希及 SSH 文档。阶段 2B 的 `enhancement.py` 以原样保留，供对照评审脚本复用。暂存清单、差异空白及常见凭据格式检查通过。历史目录、原始图、预览 PNG、权重、缓存及 logits 均保留在各自现有位置。正式运行树与资产须在服务器侧单独恢复并验收。 [预检清单](SSH_PREFLIGHT_CHECKLIST.md)

本地运行 134 项 CPU 测试，包括 19 项阶段 2C 测试、54 项阶段 2B 测试及 61 项历史退化测试。验证了 32 张锁定样本、96 个已有正式对照的 NPZ 和 metadata、补槽与裁回、增强算子以及归档完整性验证。已核对可执行 commit 的五个关键文件字节。真正的 CUDA 前向、严格 logits 配对、峰值显存和耗时等待远端有界预检。 [预检测试](test_enhancement_pilot.py) · [SSH 运行说明](SSH_PILOT_RUNNING.md)
