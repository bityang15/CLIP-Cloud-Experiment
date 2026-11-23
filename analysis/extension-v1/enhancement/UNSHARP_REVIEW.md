# 两套 Unsharp Mask 参数的 CPU 评审

采用唯一新增候选 `radius=3.0, percent=150, threshold=0`。重度模糊上，候选在四张图都产生可测的锐化，且未见明显轮廓光晕、棋盘纹或异常色块。L2 有部分像素通道达到 0 或 255，条纹样例最明显。 [逐图记录](unsharp_review.csv) · [CPU 评审脚本](review_unsharp.py)

预检使用 DTD 和 Flowers102 各两张已验证的 Base Clean 裁图。每个 L2/L3 模糊图按 V1.0 重建，8 个退化像素哈希均与正式记录一致。每张 Corrupted PNG 分别通过历史 `1.5 / 120 / 3` 和唯一候选 `3.0 / 150 / 0`，两次执行的像素逐一一致。数值均比较 Enhanced 和同一张 Corrupted。每个像素只要有一个 RGB 通道不同即记为变化。平均绝对差使用所有 RGB 通道值，单位为 8 位像素级。 [逐图记录](unsharp_review.csv)

| 等级 | 配置 | 平均变化像素 (%) | 平均绝对通道差 | 观察到的最大绝对通道差 | 单图最大新增饱和通道 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| L2 | 原配置 | 5.28 | 0.12 | 15 | 0.00 |
| L2 | 候选 | 95.81 | 4.79 | 57 | 3.85 |
| L3 | 原配置 | 0.01 | <0.01 | 4 | 0.00 |
| L3 | 候选 | 80.68 | 1.32 | 21 | 0.00 |

在原尺寸查看全部八组 `Corrupted / Original / Candidate`，L2 的花瓣和纹理边界对比度可见提升，最强条纹有轻微高光饱和。L3 保留明显模糊，候选对边缘的改变较轻。单图 L2 最大新增饱和通道率 3.85% 来自 DTD `banded_0060.jpg`，L3 的四张图均未新增饱和通道。四张 Base 裁图支持参数冻结前的代码与伪影检查。识别效果留待后续独立评测。 [逐图记录](unsharp_review.csv)

本地原尺寸对照在 `unsharp_review_preview/`，由下列命令从既有归档重建。Git 分支保留了逐图指标、两套代码和复现脚本。PNG 使用本地论文 V4 裁图及历史像素记录生成。

```powershell
& .\env\course-v1\Scripts\python.exe -B analysis/extension-v1/enhancement/review_unsharp.py
```
