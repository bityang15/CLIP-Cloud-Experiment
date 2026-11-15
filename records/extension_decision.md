# E 扩展核查记录

核查日期为 2026 年 11 月 10 日。依据《实验计划_B执行版》第 11.2 节。

## 当前验收状态

| 条件 | 状态 | 证据或待完成项 |
| --- | --- | --- |
| B 的 18 次训练、预测和基础指标 | 通过 | `B_progress.json`、`analysis_checks.json` |
| B 的温度拟合流程 | 通过 | 18 次拟合、逐样本 top-1 核查、边界命中数为零 |
| DPC 官方导入与优化器接口适配 | 真实图像预验收通过 | 独立使用 DPC 随仓库发布的 Dassl，两个数据集均通过更新与重载检查 |
| 负样本库来源及监督预算 | 通过 | 用户提供官方原包，六组种子均限制为 Base 16-shot 清单 |
| 一个种子的两阶段完整训练与评测 | 通过 | DTD seed 1 完整 CoOp → DPC、三组预测与重载通过 |
| 实测预算含 20% 余量 | 通过 | 预跑 519.23 秒，剩余保守外推含余量为 6853.78 秒 |

11 月 10 日 19 时 54 分，E 的扩展条件全部通过，开始补齐正式矩阵。DTD seed 1 预跑采用相同冻结配方、清单和代码，已计入 12 个新增正式阶段。其余 10 个阶段按相同配置执行。

新增 12 个阶段于 21 时 03 分全部完成，温度拟合、采样审计、表图生成和本地完整备份均已通过。E 的本地核查于 21 时 18 分完成。[完整实验汇报](<D:/courses/计算机视觉技术与应用/实验复现/结果_E/实验汇报.md>)

预算按两组剩余 DTD 配对使用实测耗时、三组 Flowers102 配对使用三倍 DTD 耗时估算，再乘以 1.2。得到 6853.78 秒，约 1 小时 54 分钟，可在 11 月 13 日中午前完成。[机器可读判断记录](<D:/courses/计算机视觉技术与应用/实验复现/结果_E/records/E_extension_decision.json>)

## 官方标注来源

DPC README 提供 Google Drive 与百度网盘两个官方标注下载入口。Google Drive 文件链接当前返回 404。百度网盘备用链接可验证公开提取码，分享页显示 `SPLE_database.rar`，大小 5342205 字节。用户下载后将原包放入指定目录，已完成解包和内容核查。[DPC 官方 README](https://github.com/JREion/DPC/blob/01e6b599d6515c38f4d723a8ea844ac4d1abfadd/README.md) [百度网盘官方备份，提取码 cvpr](https://pan.baidu.com/s/1LKU0g7H9o14GnZ0aZ-QMuw?pwd=cvpr) [原 Google Drive 入口](https://drive.google.com/file/d/1kMManryXLEYB6rMoeBwHiDzgmE4X_Lbz/view)

本次通过公开分享页面和正常提取码接口确认文件存在。内置浏览器的页面控制持续超时，Windows Chrome 控制因无法可靠识别当前网址而停止。随后用户自行完成官方原包下载。原包 SHA256 为 `98043c32eb8520f73a2cccecf6f1a10aaefdcca43491f4e41f1323acb768a384`。

原包包含 12 个 JSON。已提取 DTD 与 OxfordFlowers 两份，官方训练库分别含 2820、4093 张图像。两个数据集的三个种子中，所有 Base 16-shot 图像均存在于官方训练库，类别标签一致。逐种子生成受限库后，DTD 各 384 张、24 类，Flowers102 各 816 张、51 类，每类均为 16 张，与验证、测试图像的交集均为零。训练时逐个核对实际负样本路径，并记录每次扩展组的实际样本数。[标注验收](<D:/courses/计算机视觉技术与应用/实验复现/方案E/annotation_checks.json>)

## 官方适配依据

DPC 固定提交为 `01e6b599d6515c38f4d723a8ea844ac4d1abfadd`。当前公开仓库的 CoOp 分支仍引用缺失的 `trainers/base.py` 和 `trainers/optim.py`，仓库当前无对应发布包。作者在 issue 中明确建议使用 DPC 随仓库发布的修改版 Dassl。[DPC CoOp 入口](https://github.com/JREion/DPC/blob/01e6b599d6515c38f4d723a8ea844ac4d1abfadd/trainers/sple_coop.py) [作者关于 Dassl 的说明](https://github.com/JREion/DPC/issues/6#issuecomment-3245598118)

官方 DePT 仓库中可取得相应模块，固定提交为 `d4ff41684bac42da30ebb6898496772dec3157f2`。已保存 `DePT_base.py`、`DePT_optim.py` 和上游许可证至本地 `方案E/官方适配参考/`。DPC 作者的相关推理头说明也指向 DePT。[DePT 上游](https://github.com/Koorye/DePT/tree/d4ff41684bac42da30ebb6898496772dec3157f2) [作者关于 DePT 的说明](https://github.com/JREion/DPC/issues/2#issuecomment-2988544486)

静态接口核查得到以下结果。

- DePT 优化器返回 `(optimizer, infos)`，与 DPC 调用点一致。
- `STAGED_LR=False` 为默认路径。启用该分支需要额外的 `OPTIM.LR_EXP`，DPC 默认配置中缺少此项。
- DePT 的训练基类主要调整轮次日志，并导入 `utils.logger.print`，DPC 尚缺对应 logger，适配时需要明确处理。
- DPC 内置 Dassl 的 `build_trainer(cfg, name=...)` 接口与 B 安装的通用 Dassl 存在差异。E 应通过独立进程的 `PYTHONPATH` 使用内置版本，并单独记录运行环境。

## 独立适配与预验收

工作副本为 `code/DPC_expE`。保持 B 的 117 个源码文件校验值一致，复用其数据清单、指标和导出逻辑。DPC 进程单独加载内置 Dassl。将 DePT 基类中缺失的日志导入改用 Python 内置输出，优化器保留官方返回接口。

本次还处理了以下代码问题。

1. 补充本地包初始化文件，避免引用共用模块时误导入 B 的同名 `trainers` 包。首次导入失败日志已保留。
2. DPC 的 CLIP 构造缺少当前 `build_model` 所需的设计参数，改为调用同仓库 CoOp 已提供的加载函数。
3. 初始化第二阶段提示时执行独立克隆，避免 `nn.Parameter(mixed_ctx)` 与第一阶段提示共享存储。训练时逐次核对第一阶段提示校验值。
4. converse 模式仅在训练时还原并优化平行提示，评测时按配置使用融合提示。检查点加载后按 Base/New 权重构建对应提示，并重新生成当前候选类别的 token 缓冲。
5. DPC 原数据加载分支将测试集作为验证集，当前两个数据集复用 B 已验收的加载器，保持独立 Base 验证集。

两个数据集均完成每阶段一个真实图像更新、有限梯度、检查点精确重载、训练与评测提示权重核对、负样本预算检查。New 权重为零时，DPC 与对应 CoOp 的完整 New 测试 logits 逐项一致。预验收运行放在独立的 `outputs/E_preflight/attempt01`。

完整预跑采用官方 `CoOp/vit_b16.yaml` 的 200 轮第一阶段，以及 `SPLE/CoOp/vit_b16_ep20.yaml` 的 20 轮第二阶段。上下文长度均为 16、随机初始化、FP16，学习率均为 0.002。训练 batch size 分别为 32、4，Top-K 配置为 8，Base/New 融合权重为 0.5/0.0。完整预跑和剩余预算已通过，预跑已纳入 E 正式矩阵。

完整采样日志显示，官方代码查询 Top-8 候选后加入正样本，并在批次内去除重复负类，扩展组的实际图像数会变化。DTD 为 1–8 张，Flowers102 为 3–9 张。Flowers102 seeds 1、2、3 分别出现 6、9、15 个包含 9 张图像的组。当前工作副本保留此采样行为，最终汇总单列实际组大小、次数和负样本预算核查。
