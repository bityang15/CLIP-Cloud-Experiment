# CLIP 图像增强协议 V1.0

冻结于 2026-11-23。历史对照为 `formal-v1-confirmed-01`。机器可读配置与预检清单随代码一起锁定。 [增强参数](enhancement_protocol_v1.json) · [样本及正式对照哈希](pilot_inputs_v1.json)

## 处理顺序和参数

解码 RGB → 历史 Bicubic Resize → 历史 CenterCrop 224×224 → 正式 V1.0 Corruption → Enhancement → 历史 ToTensor/CLIP Normalize → 冻结模型 → FP32 raw logits。

| 退化 | 等级 | 固定增强 |
| --- | --- | --- |
| Gaussian Noise | L2、L3 | OpenCV 双边滤波，`d=5`、`sigmaColor=50`、`sigmaSpace=5`、`BORDER_REFLECT_101` |
| Gaussian Blur | L2、L3 | Pillow UnsharpMask，`radius=3.0`、`percent=150`、`threshold=0` |
| Gamma Darkening | L2、L3 | OpenCV 8 位 RGB → Lab，CLAHE 只处理 L。`clipLimit=2.0`、`tileGridSize=(8, 8)`，合并原 a/b 后转回 RGB |

增强算子在处理图像前绑定并冻结，单次调用只接收一份 224×224、`uint8` 的 Corrupted RGB。增强不使用 label、image ID、退化强度或退化生成参数。JPEG 和 L1 不在本协议范围，历史 Clean 与 Corrupted 继续沿用原协议。原 Unsharp 参数 `1.5 / 120 / 3` 作为 CPU 对照保留。唯一候选通过了实际像素变化和视觉预览检查，冻结决策未使用识别准确率、PSNR 或 SSIM。 [Unsharp 评审](UNSHARP_REVIEW.md) · [实现](enhancement_v1.py)

## SSH 预检和判定

- 每个数据集的 seed 1 正式 Base 和 New 清单各选 8 张，锁定历史首批的 0 起始位置 `0, 14, 28, 42, 56, 70, 84, 99`。32 张图及其顺序与清单 SHA-256 均有记录。四种方法均使用同一组图及历史候选类顺序。Zero-shot CLIP 是一份确定性模型，三个提示学习方法在预检中使用 seed 1。
- 每种方法及 Base/New 先用原 `course_eval.inference.export` 跑六个 Corrupted 对照。按 ID 对齐正式 raw logits 后检查标签、类序、原图/退化像素/张量哈希与冻结检查点和历史配置。对照须通过 V1.0 的严格阈值，包括最大绝对 logit 误差 `1e-5`、Top-1 一致率 `100%`，以及原 Accuracy、NLL、ECE 容差。任一硬检查失败即停止。通过后，同一组冻结模型用相同样本执行六个增强条件。
- 历史 8 图 GPU 预检的 192 个退化导出，与正式 100 槽结果按 ID 比较，只有 37 个满足 `1e-5`，全部 192 个 Top-1 一致，最大绝对差 `0.03125`。本轮预检让 8 张图保留历史槽位，其余 92 槽重复该组第一张已处理图像的张量，转发给原历史模型后只保留 8 行。对照和增强采用同一机制，历史 Corrupted 读取、退化、归一化和 raw exporter 均照旧。无法通过严格阈值时保留差异报告。
- 预期 `96 Corrupted + 96 Enhanced` 份 NPZ，每份 8 行，合计 1,536 行。GPU 实际处理 19,200 个槽位，其中包括填充副本。原图只读取 32 种身份。跨四种方法复核 192 组不同图像条件的增强 RGB 和归一化张量哈希。增强导出记录退化前后哈希、参数、图像 ID、候选类、模型、seed、配置和检查点哈希。

## 正式实验计划

后续正式评测使用 DTD 与 Flowers102 的完整 Base/New 测试集、三种退化及 L2/L3，MaPLe、MaPLe+TCPT 和 PromptSRC 的 seed 1/2/3 全覆盖。Zero-shot CLIP 每个条件一份确定性预测。计划 240 份增强预测，原正式 Corrupted 结果可作为独立配对对照。正式运行按历史正常批次顺序处理全量数据，预检的补槽针对 8 图子集。本阶段提供有界预检入口，完整运行和云端执行需后续确认。 [执行计划](SSH_PILOT_RUNNING.md)
