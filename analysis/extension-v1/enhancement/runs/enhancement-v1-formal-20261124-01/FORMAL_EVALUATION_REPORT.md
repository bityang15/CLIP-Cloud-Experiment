# 阶段 4 正式增强评测验收报告

`enhancement-v1-formal-20261124-01` 完整通过。反锐化遮罩在两数据集的 Gaussian Blur L2/L3 上均提高了 H，四种方法方向一致。CLAHE 在所有 Gamma Darkening 条件下提高了 PSNR 和 SSIM，同时四种方法的 H 均下降。双边滤波的识别收益较小，取决于数据集和强度。 [逐条件统计](condition_summary.csv)

## 范围与统计口径

使用代码 `504eeeeae1a9d357eabf4b726c0d459b182189df`、历史 100 张批次及自然尾批。DTD 与 Flowers102 的 Base/New 分别使用历史候选类。三个提示学习方法各用 seed 1/2/3，Zero-shot CLIP 使用一份确定性预测。Enhanced 与已验收的 `formal-v1-confirmed-01` Corrupted 按同一数据集、方法、seed、退化、强度及样本顺序配对。 [运行记录](RUN_RECORD.json) · [冻结协议](../../ENHANCEMENT_PROTOCOL_V1.md)

每个训练 seed 先计算 Base/New Accuracy、两者调和均值 H，以及 New ECE/NLL，再对三个 seed 求均值及样本标准差。Zero-shot CLIP 仅计一次。识别总表对 48 个数据集 × 方法 × 退化和强度单元等权平均。图像指标以历史 Clean 裁图为参照，对 24,930 个独立图像条件逐图计算和汇总。总体图像指标按这些记录等权平均。 [逐 seed 配对](paired_per_seed.csv) · [均值与标准差](paired_summary.csv) · [分组图像质量](image_quality_summary.csv)

## 总体变化

| 指标 | Corrupted | Enhanced | Enhanced − Corrupted |
| --- | ---: | ---: | ---: |
| Base Accuracy (%) | 57.09 | 59.01 | +1.93 pp |
| New Accuracy (%) | 46.53 | 48.20 | +1.67 pp |
| H (%) | 50.93 | 52.66 | +1.73 pp |
| New ECE，10 bins (%) | 12.26 | 12.53 | +0.274 pp |
| New ECE，15 bins (%) | 12.39 | 12.65 | +0.254 pp |
| New ECE，20 bins (%) | 12.60 | 12.75 | +0.153 pp |
| New NLL | 2.1510 | 2.1000 | −0.0509 |
| PSNR (dB) | 16.54 | 18.03 | +1.49 dB |
| SSIM | 0.4341 | 0.4876 | +0.0535 |

DTD 和 Flowers102 的平均 H 分别上升 0.79 和 2.67 pp。New ECE（15 bins）在 DTD 平均增加 0.77 pp，在 Flowers102 平均下降 0.26 pp。模型和 seed 层面统计共包含 249,300 个配对预测，Base 的正确→错误与错误→正确分别为 3,471 和 5,854，New 分别为 4,310 和 6,567。还有 20,923 条预测在两次皆错时改变了 Top-1 类别。 [逐 seed 错误转移](transitions_per_seed.csv) · [错误转移汇总](transitions_summary.csv)

## 分条件结果

各行识别指标对四种方法等权。`H↑方法` 统计四种方法中 seed 均值 H 上升的数量。所有差值方向为 Enhanced − Corrupted，Accuracy、H 与 ECE 单位为百分点。 [完整条件表](condition_summary.csv)

| 数据集 | 退化 | 等级 | ΔBase | ΔNew | ΔH | H↑方法 | ΔNew ECE15 | ΔNew NLL | ΔPSNR (dB) | ΔSSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DTD | Gaussian Noise | L2 | +1.24 | −0.14 | +0.34 | 3/4 | +1.44 | +0.0378 | +1.21 | +0.0405 |
| DTD | Gaussian Noise | L3 | +0.03 | −0.06 | −0.04 | 2/4 | +1.05 | +0.0454 | +0.20 | +0.0040 |
| DTD | Gaussian Blur | L2 | +5.05 | +5.49 | +5.11 | 4/4 | +1.22 | −0.1424 | +1.34 | +0.0692 |
| DTD | Gaussian Blur | L3 | +2.29 | +3.63 | +2.95 | 4/4 | −0.03 | −0.1605 | +0.30 | +0.0099 |
| DTD | Gamma Darkening | L2 | −1.63 | −1.81 | −1.81 | 0/4 | +0.30 | +0.0491 | +2.45 | +0.0047 |
| DTD | Gamma Darkening | L3 | −1.92 | −1.47 | −1.82 | 0/4 | +0.65 | +0.0504 | +2.65 | +0.1110 |
| Flowers102 | Gaussian Noise | L2 | +0.60 | +0.65 | +0.67 | 4/4 | −0.24 | −0.0265 | +1.41 | +0.0436 |
| Flowers102 | Gaussian Noise | L3 | +0.03 | +0.35 | +0.25 | 3/4 | +0.27 | +0.0010 | +0.24 | +0.0035 |
| Flowers102 | Gaussian Blur | L2 | +12.26 | +8.06 | +9.90 | 4/4 | −0.52 | −0.2857 | +1.95 | +0.0647 |
| Flowers102 | Gaussian Blur | L3 | +9.20 | +9.10 | +9.02 | 4/4 | −1.62 | −0.3032 | +0.55 | +0.0156 |
| Flowers102 | Gamma Darkening | L2 | −1.74 | −1.50 | −1.63 | 0/4 | +0.26 | +0.0646 | +2.61 | +0.0781 |
| Flowers102 | Gamma Darkening | L3 | −2.31 | −2.28 | −2.21 | 0/4 | +0.28 | +0.0590 | +2.70 | +0.1718 |

- **识别收益最一致的是反锐化遮罩。** 四个模糊条件的 H 提高 2.95–9.90 pp，New NLL 全部下降。DTD Blur L2 的 New ECE15 增加 1.22 pp，说明该条件的识别收益伴随校准变差。 [条件统计](condition_summary.csv)
- **双边滤波的收益较小且依条件而变。** Flowers102 Noise L2 的四种方法 H 均增加，宏平均为 +0.67 pp。Flowers102 L3 平均增加 0.25 pp，方法方向未完全一致。DTD L2 的 H 平均增加 0.34 pp，New Accuracy 略降，ECE 和 NLL 上升。DTD L3 的 H 几乎持平（−0.04 pp，2 升 2 降）。 [条件统计](condition_summary.csv)
- **CLAHE 在本实验中恶化识别。** H 平均下降 1.63–2.21 pp，各条件四方法方向一致，New ECE15 和 NLL 均上升。四个条件的 PSNR 同时提高 2.45–2.70 dB，SSIM 也提高。图像保真指标与识别收益在这些条件下方向相反。 [条件统计](condition_summary.csv)

## 核验、资源和备份

- 240 份 FP32 raw logits 和 249,300 行预测全部通过配对。每行的 `image_id`、标签和候选类顺序一致。增强前像素哈希匹配正式 Corrupted，24,930 个独立增强 RGB 与归一化张量签名在模型和 seed 间一致。独立 NumPy 复算 Accuracy、H、ECE 和 NLL，与正式分析表的最大绝对差为 `1.39×10⁻¹³`。 [运行记录](RUN_RECORD.json)
- 独立 CPU 质量步骤对 24,930 条记录核验了 Clean、Corrupted 与 Enhanced 像素哈希。本地重建的 24 个样例的增强像素全部逐字节匹配，质量指标最大绝对差为 `7.94×10⁻¹⁵`。全量行的统计在本地复算。 [运行记录](RUN_RECORD.json)
- GPU 运行一次成功，用时 **53 分 11 秒**，CPU 质量阶段用时 **12 分 38 秒**。PyTorch 峰值分配 **0.970 GiB**，峰值保留 **1.527 GiB**。305 次设备采样的最大板卡显存占用为 **1,998 MiB**。 [实测资源](RESOURCE_USAGE.json)
- 完整结果与日志经 SSH 三段回传，重组后的归档为 **66,830,729 字节**。归档 SHA-256 为 `66588d4d9f8729255f0a45595531df9a5a34c1d7af9b750756022559197e516e`。本地逐项校验了封存清单中的 **1,205** 个运行文件，以及 CPU 质量表、日志和传输分片。归档保存在 `env/stage4-backups/enhancement-v1-formal-20261124-01/`。 [版本及哈希记录](RUN_RECORD.json)
