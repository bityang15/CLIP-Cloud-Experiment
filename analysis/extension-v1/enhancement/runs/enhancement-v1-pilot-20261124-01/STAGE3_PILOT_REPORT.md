# 阶段 3 云端增强预检验收

预检通过。运行 `enhancement-v1-pilot-20261124-01` 于 2026-11-24 17:05–17:07（北京时间）执行，退出码为 0，进程墙钟耗时 156 秒。远端通过 GitHub HTTPS sparse clone 检出代码 `23261793e4cfcab95ab82d4c23687c255c7d3c5a`。正式参考为 `formal-v1-confirmed-01`。 [运行记录](run_record.json) · [冻结协议](../../ENHANCEMENT_PROTOCOL_V1.md)

## 远端前置检查

DTD 的 1,692 张和 Flowers102 的 2,463 张测试图均存在。32 张锁定图像的原 JPEG SHA-256 全部匹配。ViT-B/16 共享权重通过 SHA-256 核查。18 份提示学习检查点的文件大小全部匹配，其中预检使用的 6 份 seed 1 检查点通过内容 SHA-256 核查。冻结源码、历史配置、样本和类别清单及 96 份正式对照的 NPZ 与 metadata 均通过预检。 [运行记录](run_record.json)

| 环境项目 | 验收结果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 D，驱动 595.71.05，总显存 24,564 MiB |
| Python、CUDA、cuDNN | Python 3.10.22，CUDA 12.1，cuDNN 90100 |
| 关键依赖 | NumPy 1.26.4，PyTorch 2.4.1，torchvision 0.19.1，Pillow 10.4.0，SciPy 1.13.1 |
| OpenCV | 历史环境和已检查的其他环境均缺少可复用的 `cv2`。经授权在历史实验 Python 环境以 `pip install --no-deps` 增加 `opencv-python-headless==4.11.0.86` |
| 安装后检查 | `pip check` 安装前后均通过。`cv2 4.11.0`、NumPy 和 PyTorch CUDA 成功导入，已记录的其他关键依赖和 CUDA 版本一致 |

## 组合与数值验收

- 覆盖 DTD、Flowers102 各 8 张 Base 与 8 张 New。覆盖 Zero-shot CLIP、MaPLe、MaPLe+TCPT 和 PromptSRC（训练方法均为 seed 1），以及 Gaussian Noise、Gaussian Blur、Gamma Darkening 的 L2/L3。
- 96 份 Corrupted 和 96 份 Enhanced NPZ 全部存在，合计 1,536 行有限 FP32 raw logits，`image_id`、标签和类别次序与正式记录相同。保留历史 100 槽批次形状时，实际处理 19,200 个 GPU 槽位，其中包含复用的组内填充样本。
- 本地独立复算确认 96 份 Corrupted 对照、768 行的最大绝对 logit 误差为 **0.0**，冻结容差为 `1e-5`。Top-1 一致率为 **100%**，Accuracy、NLL 和三种分箱 ECE 的最大差值均为 0。
- 共核验 1,536 条输入记录。原图哈希和增强前的退化 RGB 哈希均匹配，96 份 Corrupted 的张量哈希与历史一致。192 种增强图像条件的增强 RGB 与归一化张量哈希均跨四种方法一致，增强后的像素哈希均发生变化。配置和预测文件哈希全部通过核对。 [逐组合记录](pilot_metrics.csv)

## 耗时、显存与正式资源预算

预检观测到 PyTorch 峰值分配显存 **0.975 GiB**，峰值保留显存 **1.527 GiB**，模型初始化峰值分配显存 **0.824 GiB**。两种变体的 CUDA forward 累计时间分别为 6.11 秒与 5.78 秒，导出墙钟之和分别为 42.24 秒与 41.97 秒。进程墙钟还包含前置检查、模型初始化、校验和文件清单生成。 [逐组合记录](pilot_metrics.csv) · [资源预算](RESOURCE_BUDGET.json)

| 正式运行方案 | 新导出份数 | 预计墙钟 | 预计输出 |
| --- | ---: | ---: | ---: |
| 全量 Enhanced，复用已验收的 Corrupted | 240 | 53.4 分钟 | 489.8 MiB |
| 全量 Enhanced 并重跑配对 Corrupted | 480 | 97.3 分钟 | 956.9 MiB |

每种变体估计覆盖 249,300 个预测行。预算包含三个提示学习方法的三个训练 seed，以及一份确定性 Zero-shot CLIP。估算将 100 槽 CUDA 计时按正式批次数外推，将预处理和传输时间按图像数外推，并计入一次模型初始化。大批次的吞吐、缓存命中和文件压缩率会影响正式耗时与空间。显存数字为此次预检实测。 [资源预算](RESOURCE_BUDGET.json)

## 本地备份验收

通过 SSH 回传压缩包及远端 SHA-256 记录。本地确认压缩包为 622,715 字节，逐项复核清单中的 869 个运行文件，以及外部日志和前置检查报告。原始 NPZ、完整日志和远端清单保存在本地 `env/stage3-backups/enhancement-v1-pilot-20261124-01/`。GitHub 同步本报告、逐组合时序、资源预算及配置/版本记录。发布的 CSV 使用 LF 行尾，原始 CSV 保留在备份中。 [运行记录](run_record.json)

| 对象 | SHA-256 |
| --- | --- |
| 本地结果压缩包 | `6f3eba5e77ec0229b17475f9dc8d1d17ae950daccd604967ddd7c00389d46276` |
| 运行文件清单 | `589a6da4c9fbe66595ace4725014e6e6b72695c02c90b81c304aebeb6e5dc93d` |
