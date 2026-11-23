# RTX 4090 云端环境预检查清单

本清单供用户确认云端预检后使用，本轮未连接租用服务器。自动检查入口是 `pilot.py preflight`，它不构建模型。 [运行步骤](SSH_PILOT_RUNNING.md)

| 项目 | 验收要求 |
| --- | --- |
| SSH 与 Git | 登录已租用实例并核对主机密钥。GitHub checkout 指向已确认的 40 位代码 commit，运行时源码和两份 JSON 与该 commit 的字节一致。 |
| GPU、驱动和 CUDA | `nvidia-smi` 可用，记录驱动、总显存与可用显存。所选设备是 RTX 4090，PyTorch 可访问，CUDA runtime 12.1 和 cuDNN 90100 与正式记录一致。 |
| 依赖 | Python 3.10.x，正式 V1.0 包版本，`opencv-python-headless==4.11.0.86`，Pillow 的 libjpeg 6.2、libjpeg-turbo 3.0.3。检查历史 trainer 模块导入以及四方法配置读取。缺失时列出具体包和版本。 |
| 历史源码 | `course_eval` 和归档 trainer 源码逐文件匹配正式 `run_manifest.json` 的 SHA-256。运行旧代码时使用 `-B`，避免在旧运行树写字节码。 |
| 原始测试图 | DTD `datasets/dtd/images` 和 Flowers102 `datasets/oxford_flowers/jpg` 下全量正式测试图文件均存在。按历史三 seed 清单核查覆盖，对 32 张锁定预检图另验原 JPEG SHA-256。 |
| 共享 CLIP 权重 | `weights/ViT-B-16.pt` 的 SHA-256 与 `records/config_manifest.json` 相符。旧 harness 直接读取该文件。 |
| 提示学习权重 | 用 `records/checkpoints.json` 核对两数据集、三方法、三 seed 的 18 份归档路径和字节数。预检要加载的 6 份 seed 1 文件核对 SHA-256，正式前再校验其余 12 份。 |
| 对照与历史数据划分 | 正式 run manifest、96 项对照的 metadata/NPZ、样本及类别清单均匹配锁定哈希。冻结 Batch Size 为 100。原始对照与增强输出文件使用独立路径。 |
| 输出空间与备份 | 在可写持久盘创建新输出目录，给全部 192 份预检 NPZ、日志和 SHA-256 清单留足空间。在正式前按 `budget.json` 核对预计空间，预留一份本地备份。 |

本地已有的归档 `env/course_v1_release_ready.tar.gz` 的 SHA-256 为 `657811f1228a19eebb332dfa0371aa504aaa99722859efbd9f288ec21d6c4b1b`。正式结果备份 `env/formal-v1-confirmed-01-backup01.tar.gz` 的验收 SHA-256 为 `3a42bd8478552a1c2467472d07aad732f03295b65bcc09eb20dd01c459c174ae`。两个归档分开保存，按确认后需要通过已有 SSH 传输。完整原始图、ViT-B/16 共享权重及提示检查点需要另用受控资产来源。 [冻结输入锁](pilot_inputs_v1.json)
