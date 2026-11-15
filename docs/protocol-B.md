# CLIP 提示学习实验正式执行计划

## 0 执行摘要

方案于 2026 年 11 月 10 日确定。先完成方案 B，运行顺利且时间充足时扩展至 E。所有日期均采用北京时间。

| 项目 | 正式安排 |
| --- | --- |
| 主结果三个方法 | MaPLe、PromptSRC、MaPLe+TCPT |
| 数据集 | DTD、Flowers102 |
| 协议 | CLIP ViT-B/16，16-shot Base-to-New |
| 种子 | 1、2、3 |
| 正式训练 | 18 次 |
| 原始测试预测 | 36 份，分别保存 Base 与 New |
| 校准验证预测 | 18 份，仅使用 Base 验证集 |
| 温度拟合 | 18 次，每个已训练模型对应一个温度 |
| 冻结 CLIP 参照 | 两数据集各评测 Base/New，共 4 份预测 |
| 正式结果快照 | 11 月 13 日周五中午 12 点 |
| PPT、口播和证据包完成 | 11 月 14 日周六晚 |
| 排练与核查 | 11 月 15 日周日 |
| 现场汇报 | 11 月 17 日周二 |

当前应从第 4 节的数据与权重准备开始。云端环境已有检查记录。第 5 节的统一训练开关和预测导出接口需要先完成接入与验收，第 7 节命令模板在该阶段通过后使用。

## 1 实验要回答的问题

汇报主问题为少样本适配之后，新类别识别能力与预测概率可靠性如何共同变化。

| 问题 | 对照 | 重点指标与可形成的证据 |
| --- | --- | --- |
| 不同提示方法的泛化与校准表现如何 | MaPLe 与 PromptSRC | Base、New、H，以及两组 ECE/NLL |
| 训练期校准约束带来什么变化 | 同一训练器下的 MaPLe 与 MaPLe+TCPT | 相同种子的准确率与概率质量差值 |
| 简单后处理能改善多少，能否迁移到新类别 | 每个主方法的原始输出与温度缩放输出 | Base 验证拟合后，Base/New 的 ECE、NLL 变化 |

PromptSRC 与 MaPLe 使用各自明确记录的配方，比较方法整体表现。TCPT 与 MaPLe 共用结构、初始化、数据顺序和训练配方，通过正则开关形成成对对照。

TCPT 主文已经包含 Temperature Scaling 对照。方案 B 的定位为统一协议下的部分复现与补充分析，增加 NLL、固定检查点配对、三个种子波动和分箱敏感性记录。[TCPT 论文](https://arxiv.org/abs/2602.19024)

## 2 当前云端基础与目录

### 2.1 已有检查记录

11 月 10 日 13 时 46 分保存的记录显示，云端为 Ubuntu 22.04、RTX 4090 D 24GB、18 核 CPU、60GiB 容器内存、50GB 数据盘。独立环境为 Python 3.10.22、PyTorch 2.4.1、torchvision 0.19.1，PyTorch CUDA 运行库为 12.1。

依赖一致性、两份官方源码训练入口导入、合成输入的 FP16 前向反向与 CUDA 算子检查已通过。真实图像数据训练、正则梯度和最终预测一致性按后续阶段验收。[云端配置记录](environment.md)

### 2.2 目录约定

以下终端命令均在云服务器的 Linux 终端中执行。云端根目录统一为 `/root/autodl-tmp/clip-exp`。

```text
/root/autodl-tmp/clip-exp/
  env/                         已有 Python 环境
  code/PromptSRC/               已归档的作者源码
  code/TCPT/                    已归档的作者源码
  code/Dassl.pytorch/            已固定的框架版本
  code/PromptSRC_expB/           本轮工作副本，阶段 2 建立
  datasets/                     共用图像、划分与少样本缓存
  weights/                      预训练 CLIP 权重
  records/                      环境、来源、数据清单和验收记录
  scripts/                      已有环境脚本及后续接入工具
  outputs/B/                    方案 B 全部正式输出
  outputs/E/                    通过扩展条件后的 DPC 路线输出
  downloads/                    下载的原始数据文件
```

单次正式运行使用独立目录，如 `outputs/B/dtd/maple/seed1/attempt01`。配置修订后递增 attempt 编号，汇总表记录实际采用的运行标识。

## 3 固定实验规格

### 3.1 数据与监督

| 项目 | DTD | Flowers102 |
| --- | --- | --- |
| 配置名 | `dtd` | `oxford_flowers` |
| Base 类数 | 24 | 51 |
| New 类数 | 23 | 51 |
| Base 训练图像，每类 16 张 | 预计 384 张 | 预计 816 张 |
| Base 校准验证图像，每类最多 4 张 | 预计 96 张 | 预计 204 张 |

以上图像数需要核对实际清单。三种方法在相同数据集、种子下共用训练清单与 Base 验证清单。沿用官方先少样本抽样、后筛选类别组的顺序，共用同一数据根目录和缓存。

训练监督来自 Base 类训练标签。温度拟合使用独立的 Base 验证标签，实际使用量单列。New 标签用于最终评测和预先规定的诊断。超参数、温度和检查点选择依据训练数据、Base 验证数据及本计划固定规则。

测试分别给定 Base、New 候选类别集合，记录类别名、顺序和重映射标签。图像输入为 224 × 224，训练增强和测试预处理采用同一套已核验配置。

### 3.2 主方法配方

| 参数 | MaPLe | PromptSRC | MaPLe+TCPT |
| --- | --- | --- | --- |
| 轮数 | 5 | 20 | 5 |
| 初始学习率 | 0.0035 | 0.0025 | 0.0035 |
| 训练 batch size | 4 | 4 | 4 |
| 优化器、调度 | SGD、余弦调度 | SGD、余弦调度 | SGD、余弦调度 |
| 提示深度 | 9 | 视觉、文本均为 9 | 9 |
| 上下文长度 | 2 | 视觉、文本均为 4 | 2 |
| 初始化文本 | `a photo of a` | `a photo of a` | `a photo of a` |
| 精度入口 | fp16 | fp16 | fp16 |
| 最终权重 | 第 5 轮 | 第 20 轮的 GPA 聚合权重 | 第 5 轮 |

CLIP 主体冻结，实际可训练参数名与数量保存。TCPT 正则统计、离线指标与保存的 logits 使用 FP32，温度搜索的 NLL 累积可使用 FP64。

PromptSRC 采用文本损失权重 25、图像损失权重 10、GPA 均值 15、标准差 1，以及官方输出一致性和文本多样性设置。其训练成本按实际轮数和时间报告。

### 3.3 TCPT 的目标与冻结参照

`L = CE - 0.1 × mean(m) + 0.01 × Var(m) + 5.0 × L_mom`

- m 为真实类别 logit 与最强错误类别 logit 的差。
- 方差采用总体形式 `unbiased=False`。
- `L_mom` 匹配归一化文本特征的均值和协方差，统计范围为当前 Base 候选类别。
- 协方差按参与统计的类别数归一化。冻结参照以评测模式生成并缓存。
- DTD 模板为 `{} texture.`。
- Flowers102 模板为 `a photo of a {}, a type of flower.`。

这套参数取自本地 TCPT 归档的实际损失入口。实现记录交代其与论文通用训练描述的对应关系。

## 4 阶段 1 数据、权重与清单

### 4.1 进入环境并确认资源

```bash
cd /root/autodl-tmp/clip-exp
export BROOT=/root/autodl-tmp/clip-exp
export PYTHON_BIN="$BROOT/env/bin/python"
nvidia-smi
df -h "$BROOT"
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" "$BROOT/scripts/verify_environment.py"
```

将本次检查时间和输出保存至 `records`。环境变更后再次记录版本与原因。

### 4.2 准备数据

先盘点已有文件，再执行缺少部分的下载。首次准备所需文件如下，链接来自已归档的作者数据说明。[作者数据说明](../code/PromptSRC_expB/docs/DATASETS.md)

| 数据文件 | 来源 |
| --- | --- |
| DTD 图像 | [Oxford DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz) |
| DTD 固定划分 | [作者划分 JSON](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view) |
| Flowers102 图像 | [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) |
| Flowers102 标签 | [imagelabels.mat](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat) |
| Flowers102 类别名 | [cat_to_name.json](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view) |
| Flowers102 固定划分 | [作者划分 JSON](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view) |

首次下载示例使用已有的 gdown 环境。

```bash
mkdir -p "$BROOT/downloads" "$BROOT/datasets/oxford_flowers" "$BROOT/weights"
curl --fail --location --retry 3 \
  'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz' \
  --output "$BROOT/downloads/dtd-r1.0.1.tar.gz"
curl --fail --location --retry 3 \
  'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz' \
  --output "$BROOT/downloads/102flowers.tgz"
curl --fail --location --retry 3 \
  'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat' \
  --output "$BROOT/datasets/oxford_flowers/imagelabels.mat"
tar -xzf "$BROOT/downloads/dtd-r1.0.1.tar.gz" -C "$BROOT/datasets"
tar -xzf "$BROOT/downloads/102flowers.tgz" -C "$BROOT/datasets/oxford_flowers"
"$PYTHON_BIN" -m gdown '1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x' \
  -O "$BROOT/datasets/dtd/split_zhou_DescribableTextures.json"
"$PYTHON_BIN" -m gdown '1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0' \
  -O "$BROOT/datasets/oxford_flowers/cat_to_name.json"
"$PYTHON_BIN" -m gdown '1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT' \
  -O "$BROOT/datasets/oxford_flowers/split_zhou_OxfordFlowers.json"
```

网络受限时，可以按上述来源下载后上传相同文件。对 JSON 进行解析并记录来源、大小和 SHA256，再冻结正式划分。

正确布局如下。

```text
datasets/dtd/images/<texture_class>/*.jpg
datasets/dtd/split_zhou_DescribableTextures.json
datasets/oxford_flowers/jpg/image_00001.jpg
datasets/oxford_flowers/imagelabels.mat
datasets/oxford_flowers/cat_to_name.json
datasets/oxford_flowers/split_zhou_OxfordFlowers.json
```

### 4.3 准备共同的 CLIP 权重

使用归档 CLIP 代码中 ViT-B/16 对应的官方权重。

```bash
curl --fail --location --retry 3 \
  'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt' \
  --output "$BROOT/weights/ViT-B-16.pt"
sha256sum "$BROOT/weights/ViT-B-16.pt"
```

预期 SHA256 为 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`。将校验输出存入 `records/weight_sha256.txt`。

归档加载器默认从 `/root/.cache/clip/ViT-B-16.pt` 读取。缓存命中时核对同一校验值，缓存缺失时建立指向本轮权重的符号链接。

```bash
mkdir -p /root/.cache/clip
if [ ! -e /root/.cache/clip/ViT-B-16.pt ]; then
  ln -s "$BROOT/weights/ViT-B-16.pt" /root/.cache/clip/ViT-B-16.pt
fi
sha256sum /root/.cache/clip/ViT-B-16.pt
```

### 4.4 数据验收清单

- [ ] 两份划分 JSON、类别名文件均能解析，划分中所有图像路径存在。
- [ ] 每个数据集的 train、val、test 图像身份交集核对通过。
- [ ] DTD 类数为 47，Flowers102 为 102，Base/New 类别顺序与官方筛选逻辑一致。
- [ ] 保存三种子的实际 Base 训练和验证清单，核对每类训练 16 张、验证最多 4 张。
- [ ] 同数据集、种子下，三方法使用同一份清单和校验值。
- [ ] 测试样本和类别映射在种子间一致，Base/New 单独记录。
- [ ] 生成 `records/data_manifest.json`、`records/data_checks.json` 和逐种子清单。

预计耗时为 1–3 小时，受下载速度影响。通过数据验收后进入阶段 2。

## 5 阶段 2 统一训练与预测接口接入

### 5.1 本阶段需要交付的工作

当前官方代码提供训练器，本轮还需要接入以下实验接口。它们属于正式训练前的实现任务。

1. 建立 `code/PromptSRC_expB` 工作副本，以 PromptSRC 归档的框架、数据和 MaPLe/PromptSRC 训练器为基础。
2. 在同一 MaPLe 工作训练器中接入已核对的 TCPT 两项正则，以一个明确开关控制开、关。
3. 在统一评测路径中导出图像标识、标签、类别顺序和原始 logits。
4. 增加离线指标、温度网格拟合和汇总入口，复用已保存的预测。
5. 保存完整工程差异、接口验收结果和固定配置。

工作副本相对原始归档的修改保存在 `records/implementation.patch`，最终配置写入 `records/config_manifest.json`。每次运行保存完整展开后的配置。

### 5.2 新接口约定

以下为需要在工作副本注册并实现的配置项，第 7 节命令模板使用这些名称。

| 配置项 | 用途 |
| --- | --- |
| `TRAINER.MAPLE.USE_TCPT` | False 使用 MaPLe 基线损失，True 加入固定配方的两项正则 |
| `EXPERIMENT.EXPORT_DIR` | 评测时保存预测文件的目录，空字符串表示常规评测 |

MaPLe 基线与 TCPT 的网络结构、初始化、数据顺序、学习率、轮数和精度统一。冻结文本特征以评测模式缓存。正则统计使用 FP32，并保留对可训练提示的梯度。

配对验收时记录初始化参数校验值和前若干批次的图像标识。冻结参照的构建与缓存也纳入随机状态管理，确保同种子数据顺序对应。

需要特别处理的已知事项如下。

| 已核对事项 | 接入时的处理 |
| --- | --- |
| MaPLe 配置文件名含 ep5，实际 `MAX_EPOCH` 为 2 | 配置与运行参数显式固定为 5 |
| TCPT 归档部分精度分支损失不同 | 对采用的 fp16 入口逐项检查 CE、间隔、矩匹配和梯度 |
| TCPT 原配置含额外弯引号 | 初始化文本统一为 `a photo of a` |
| PromptSRC 训练后使用 GPA | 对聚合权重、保存和重新加载的预测做一致性检查 |
| Base/New 类数不同 | 重载时按官方规则处理类别 token 缓冲，核对提示参数及候选类映射 |

### 5.3 预测文件格式

每次评测写出 `predictions.npz` 和 `metadata.json`。

| 字段 | 内容 |
| --- | --- |
| `logits` | FP32，形状 N×C，保留 CLIP 原始 logit scale |
| `labels` | 当前候选集合中的 0 起始整数标签，长度 N |
| `image_ids` | 相对于数据根目录的图像标识，长度 N |
| `classnames` | 与 logits 列一一对应的类别名，长度 C |
| metadata | 数据集、方法、种子、运行标识、候选组、split、检查点和配置/数据校验值 |

`test_base`、`test_new`、`val_base` 分别建目录。验证和测试均使用确定性的评测预处理。

方法标识使用 `maple`、`promptsrc`、`tcpt`。MaPLe 与 TCPT 共用 MaPLe 训练器，元数据同时记录正则开关和配置校验值，以区分两种正式配置。

### 5.4 启动前验收

- [ ] 开关关闭时，相同固定批次的前向和 CE 与 MaPLe 基线对齐。
- [ ] 两项 TCPT 正则实际进入总损失，梯度有限且影响提示参数。
- [ ] 冻结编码器与冻结参照处于预定状态，可训练参数名和数量符合配置。
- [ ] 三种方法各完成一个真实数据 batch 的前向、反向与参数更新。
- [ ] Base、New 各完成一次预测导出，类别列与重映射标签对应。
- [ ] 离线准确率、ECE、NLL 与在线结果一致。
- [ ] PromptSRC GPA 权重完成保存与重载核查。
- [ ] 新配置项已注册，训练器能通过正常 `train.py` 入口执行。
- [ ] 记录通过结果至 `records/preflight.json`，冻结正式配方。

预计人工工作为 3–5 小时，首次真实数据验收另留 1–2 小时。验收失败时，先修正对应实现并记录受影响的运行，再启动后续正式任务。

## 6 阶段 3 正式训练顺序与资源记录

### 6.1 第一轮六次运行

按下列顺序完成 seed 1。冻结配方后通过核查的运行计入正式 18 次。

| 顺序 | 数据集 | 方法 | 轮数 |
| --- | --- | --- | --- |
| 1 | DTD | MaPLe | 5 |
| 2 | DTD | MaPLe+TCPT | 5 |
| 3 | DTD | PromptSRC | 20 |
| 4 | Flowers102 | MaPLe | 5 |
| 5 | Flowers102 | MaPLe+TCPT | 5 |
| 6 | Flowers102 | PromptSRC | 20 |

每次结束后导出 Base 测试、New 测试和 Base 验证预测，重载检查点复核结果。记录单轮耗时、稳定阶段秒/更新、初始化和评测时间。

### 6.2 补齐正式矩阵

完成六次 seed 1 核查后，以相同顺序补齐 seed 2，再补 seed 3，总量为 18 次。

| 配置 | DTD 运行 | Flowers102 运行 | 合计 |
| --- | --- | --- | --- |
| MaPLe | 3 | 3 | 6 |
| PromptSRC | 3 | 3 | 6 |
| MaPLe+TCPT | 3 | 3 | 6 |
| 合计 | 9 | 9 | 18 |

使用单张卡顺序运行。长任务放在 screen 或 tmux 会话内，并保存标准输出、标准错误、退出码和运行状态。

当前环境曾使用 screen，可在终端创建本轮会话，再在会话中设置变量并执行训练。

```bash
screen -S clip-b
```

会话中按 Ctrl+A，再按 D 可断开显示。重新登录后使用 `screen -r clip-b` 回到会话。执行期间检查运行日志与 GPU 状态。

恢复训练时使用同一配置及已核对的恢复点，记录恢复前后标识和随机状态。配置变更后的运行采用新的 attempt 目录。最终汇总使用通过核查的唯一运行标识。

### 6.3 资源与时间估计

当前 RTX 4090 D 的首轮实测用于更新以下排期区间。

| 工作 | 估计 |
| --- | --- |
| 18 次主训练 | 约 2.5–8.25 GPU 小时 |
| 连同评测、初始化和小规模重跑预留 | B 共约 3–13 GPU 小时 |
| Base 验证预测导出 | 约 10–45 分钟，计入上述预留 |
| 18 次温度拟合，已有 logits | 约 1–10 分钟 CPU 时间 |
| 全部 B 人工配置、分析与实验章节整理 | 约 12–21 小时 |

训练与人工整理可交叠。环境阶段已有记录，按实际完成程度扣减对应准备工时。下载、数据读取和调试进展通过日志更新排期。

## 7 接入完成后的训练和导出命令模板

本节以 `dtd`、`seed1`、`attempt01` 为例。运行前确认第 5 节验收通过，并将数据集、方法、种子与输出目录逐项对应。每个新终端重新设置下列变量。

```bash
export BROOT=/root/autodl-tmp/clip-exp
export PYTHON_BIN="$BROOT/env/bin/python"
cd "$BROOT/code/PromptSRC_expB"
```

### 7.1 MaPLe

```bash
export RUN_DIR="$BROOT/outputs/B/dtd/maple/seed1/attempt01"
mkdir -p "$RUN_DIR"
set -o pipefail
"$PYTHON_BIN" train.py \
  --root "$BROOT/datasets" \
  --seed 1 --trainer MaPLe \
  --dataset-config-file configs/datasets/dtd.yaml \
  --config-file configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml \
  --output-dir "$RUN_DIR" \
  DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
  OPTIM.MAX_EPOCH 5 TRAINER.MAPLE.USE_TCPT False \
  2>&1 | tee "$RUN_DIR/console.log"
run_status=${PIPESTATUS[0]}
printf '%s\n' "$run_status" > "$RUN_DIR/exit_code.txt"
```

其余训练采用相同的日志与退出码保存方式。退出码为 0 后，继续核对日志、检查点和预测验收结果。

### 7.2 MaPLe+TCPT

```bash
"$PYTHON_BIN" train.py \
  --root "$BROOT/datasets" \
  --seed 1 --trainer MaPLe \
  --dataset-config-file configs/datasets/dtd.yaml \
  --config-file configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml \
  --output-dir "$BROOT/outputs/B/dtd/tcpt/seed1/attempt01" \
  DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
  OPTIM.MAX_EPOCH 5 TRAINER.MAPLE.USE_TCPT True
```

### 7.3 PromptSRC

```bash
"$PYTHON_BIN" train.py \
  --root "$BROOT/datasets" \
  --seed 1 --trainer PromptSRC \
  --dataset-config-file configs/datasets/dtd.yaml \
  --config-file configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml \
  --output-dir "$BROOT/outputs/B/dtd/promptsrc/seed1/attempt01" \
  DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base
```

### 7.4 导出 Base 测试、New 测试、Base 验证

以下以 MaPLe 的第 5 轮为例。模型、配置、种子、检查点与训练时一致。MaPLe+TCPT 对应开关为 True，PromptSRC 使用其训练器、配置与第 20 轮 GPA 检查点。

```bash
export RUN_DIR="$BROOT/outputs/B/dtd/maple/seed1/attempt01"
"$PYTHON_BIN" train.py \
  --root "$BROOT/datasets" \
  --seed 1 --trainer MaPLe \
  --dataset-config-file configs/datasets/dtd.yaml \
  --config-file configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml \
  --model-dir "$RUN_DIR" --load-epoch 5 --eval-only \
  --output-dir "$RUN_DIR/eval_test_base" \
  DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
  TEST.SPLIT test OPTIM.MAX_EPOCH 5 TRAINER.MAPLE.USE_TCPT False \
  EXPERIMENT.EXPORT_DIR "$RUN_DIR/predictions/test_base"
```

New 测试将类别组改为 `DATASET.SUBSAMPLE_CLASSES new`，split 使用 `TEST.SPLIT test`，输出目录分别为 `eval_test_new` 和 `predictions/test_new`。

Base 验证将类别组设为 base、split 设为 `TEST.SPLIT val`，输出目录分别为 `eval_val_base` 和 `predictions/val_base`。

Flowers102 将数据配置替换为 `configs/datasets/oxford_flowers.yaml`，输出路径的数据集名替换为 `oxford_flowers`。seed 2、3 同步修改 `--seed` 与路径。

## 8 阶段 4 冻结 CLIP 参照

在同一数据集、相同候选类别和测试预处理下，使用第 3.3 节的单模板输出冻结 CLIP 预测。该模板选择写入配置。

分别保存 DTD 的 Base/New、Flowers102 的 Base/New，共 4 份预测。每份都按第 5.3 节格式导出。

冻结 CLIP 使用确定性评测，报告单值。训练方法按三个种子报告均值和标准差。参照表和三方法主结果表分别标注用途。

## 9 阶段 5 温度拟合与离线评测

### 9.1 每份模型的温度拟合

每次使用该模型自己的 `val_base/predictions.npz`。模型权重固定，使用验证集平均 NLL 搜索正温度 T。

1. 在 `[0.05, 20]` 上取 401 个对数等距候选温度，将最接近 1 的网格点显式设为 1。
2. 对每个候选计算稳定的 `log_softmax(logits/T)` 和平均 NLL。
3. 选择验证 NLL 最小的候选值，数值相等时选择最接近 1 的值。
4. 保存 T、验证 NLL 前后值、样本数、候选范围、边界命中情况、预测和配置校验值。
5. 将同一个 T 应用于该模型的 Base、New 测试 logits，计算缩放后的 ECE/NLL。
6. 逐样本核对正数缩放前后的 top-1 类别，并保存核查结果。

温度范围、网格、选择准则在正式搜索前固定。若最优值命中边界，报告边界命中，并保留预定主结果。

本阶段形成 18 份温度记录。经典实现可作为接入参考，本计划采用上述固定网格和 Base 验证协议。[温度缩放参考实现](https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py)

### 9.2 原始输出与缩放输出的指标

- Accuracy 为当前候选集合中的 top-1 准确率。
- 每数据集、方法、种子计算 `H = 2 × Base × New / (Base + New)`，再汇总种子。
- ECE 主结果采用 10 个等宽置信度箱，区间右闭，置信度 0 计入第一箱。
- 补充 15 箱和 20 箱 ECE，复用同一预测。
- NLL 使用原始 logits 或缩放 logits 的稳定 log-softmax 计算。
- Base 与 New 分别统计，记录每箱样本数、准确率、平均置信度。
- 三个种子报告均值及样本标准差，标准差采用 `ddof=1`。
- 成对变化按同一数据集、同一种子相减，保存全部三次差值。
- 可靠性图固定使用 seed 1，每数据集各展示 Base、New 的重点比较。

结果解释同时观察 ECE、NLL 和准确率。ECE 与 NLL 方向不同时，展示对应可靠性图、置信度和错误预测分布。原始结果与后处理结果分别成表。

## 10 阶段 6 汇总表、图与证据包

### 10.1 需要交付的表

| 文件建议名 | 内容 | 预计行数 |
| --- | --- | --- |
| `runs_raw.csv` | 数据集、方法、种子、运行标识、B/N/H、两组 ECE/NLL、资源 | 18 |
| `summary_raw.csv` | 每数据集、方法的均值与标准差 | 6 |
| `temperature_runs.csv` | 逐模型 T、验证 NLL、缩放后测试指标与核查状态 | 18 |
| `summary_temperature.csv` | 每数据集、方法的温度缩放后均值、标准差与原始差值 | 6 |
| `paired_tcpt_vs_maple.csv` | 同种子 TCPT 相对 MaPLe 的原始与缩放后差值 | 6 |
| `clip_reference.csv` | 两数据集、两候选组的冻结 CLIP 指标 | 4 |

CSV 行的纳入条件为完整预测、检查点、配置和数据清单核查通过。实际完成范围和失败运行另列状态记录。

### 10.2 需要交付的图

- Base、New、H 的逐数据集比较，显示三个种子的波动。
- 原始与温度缩放后的 New ECE、New NLL 比较。
- seed 1 的可靠性图，并保留每箱样本量。
- 训练时间、推理时间、温度拟合时间及峰值显存的成本表。

图表文字明确候选类别组、数据集、方法、种子范围和单位。论文摘录与实测结果分别标注来源。

### 10.3 证据保存清单

- [ ] 环境版本、源码来源与工作副本差异。
- [ ] 预训练权重、划分 JSON、样本清单及类别映射的校验值。
- [ ] 18 次训练的完整配置、日志、检查点、完成状态与资源记录。
- [ ] 36 份原始测试预测、18 份 Base 验证预测、18 份温度记录。
- [ ] 4 份冻结 CLIP 参照预测。
- [ ] 在线离线一致性、正则梯度、GPA 重载和温度预测排序核查。
- [ ] 汇总脚本、表格、图及最终采用的运行标识。

云端证据导回本地的 `实验复现/结果_B/`，优先保存配置、日志、预测、指标和图表，并保存最终检查点。日常阶段性备份记录文件大小和校验值。

## 11 时间安排与 E 扩展条件

### 11.1 正式排期

| 时间 | 操作目标 | 通过或交付标准 |
| --- | --- | --- |
| 11 月 10 日，周二 | 数据、权重、清单和统一入口准备 | 阶段 1、2 的验收项有记录，启动首轮 |
| 11 月 11 日，周三 | 完成 seed 1 核查，补齐 B 主训练 | 18 次训练及预测尽早形成完整 B 结果，开始温度拟合 |
| 11 月 11 日晚 20 点 | E 扩展判断 | 按第 11.2 节核查实现、证据和剩余时间 |
| 11 月 12 日，周四 | 完成 B 温度拟合、指标、图，符合条件时扩展 E | B 的汇总表可生成，E 运行状态持续记录 |
| 11 月 13 日，周五中午 12 点 | 冻结本轮正式结果快照 | 汇报采用的运行、覆盖范围、表图来源明确 |
| 11 月 13 日下午至 14 日 | 完成 PPT、口播、报告实验部分与核对 | 周六晚完成文件和一次全流程试讲 |
| 11 月 15 日，周日 | 排练与答辩准备 | 时间、来源、方法边界和结果解释核查通过 |
| 11 月 17 日，周二 | 汇报 | 使用冻结后的图表与结论 |

周三的时间点为本计划的工程选择，用于给数据整理和 PPT 留出完整时间。实际运行若需要更长时间，以 B 的正式结果和周五的汇总节点优先安排。

### 11.2 扩展 E 的全部条件

- [ ] B 的 18 次正式训练、预测和基础指标已经通过核查。
- [ ] B 的温度拟合流程已经验收，并可从保存的预测离线完成。
- [ ] DPC 官方工作副本的导入与优化器接口已适配，修改和来源已归档。
- [ ] DPC 负样本库来源可核对，符合本轮 Base 训练清单和监督预算。
- [ ] 一个种子的 DPC 两阶段完整训练与 Base/New 评测已跑通，检查点和类别映射正确。
- [ ] 实测外推的剩余训练、评测及核查时间，加 20% 余量，能在周五中午前完成。

扩展选择依据实现和数据核查状态、实际完成时间。判断记录写入 `records/extension_decision.md`，包括通过项、待处理项和实测预算。

若按 B 完成汇报，三个主方法为 MaPLe、PromptSRC、MaPLe+TCPT，DPC 在综述中使用原论文证据及机制分析。

### 11.3 E 的新增工作

| 新增配置 | 数据集、种子组合 | 新增阶段训练 |
| --- | --- | --- |
| CoOp 第一阶段及 DPC 配对参照 | 2 数据集 × 3 种子 | 6 |
| DPC（CoOp）第二阶段 | 2 数据集 × 3 种子 | 6 |
| 合计 | | 12 |

E 共 30 个阶段训练任务，复用 B 的 18 次训练。CoOp 与 DPC 的对应温度拟合各增加 6 次，连同 B 共保存 30 份温度记录，以支持相同后处理条件下的配对比较。

E 的主结果展示 PromptSRC、DPC（CoOp）、TCPT（MaPLe）。CoOp 和 MaPLe 分别进入 DPC、TCPT 的配对分析表。新增原始测试预测 24 份，Base 验证预测 12 份。

DPC 的第一、第二阶段统一上下文长度、数据清单和检查点。核对困难负类来源、实际 Top-K、推理组合权重及分组条件。当前官方 CoOp 分支引用缺失的 `.base` 与 `.optim`，且优化器接口返回值需要适配，这些属于扩展前的工程工作。[官方 DPC 入口](https://github.com/JREion/DPC/blob/main/trainers/sple_coop.py)

符合数据和配置要求的预跑可计入正式任务，记录其纳入依据。配方修订后的影响范围和重跑任务单独登记。

## 12 PPT 的结果组织与结论依据

方法部分讲 PromptSRC 的知识保持、DPC 的提示分工、TCPT 的校准约束。详细量化结果集中到方法之后的实验与分析部分。

| 正文内容 | 使用证据 |
| --- | --- |
| 三篇论文原文证据摘要 | 各自论文内部对照，标注数据范围与来源 |
| 本轮统一设置 | 数据、候选组、16-shot、种子、训练和验证标签预算 |
| 本轮识别结果 | 逐数据集 B/N/H 与三个种子 |
| 本轮校准结果 | 原始和温度缩放后的 ECE/NLL、可靠性图 |
| 配对与成本分析 | TCPT 对 MaPLe，扩 E 时加入 DPC 对 CoOp |
| 总结 | 按实际观察说明收益、波动、成本和适用范围 |

若温度缩放接近或优于 TCPT，结合准确率、New NLL 和成本讨论。若方法提高 H 同时增加 ECE，分别解释识别变化与概率质量。若种子间方向变化较大，保留三次差值和实际覆盖范围。

汇报前核对以下口径。

- PromptSRC 与 MaPLe 采用各自配方，单模块归因由相应消融提供证据。
- 本轮实验覆盖两个数据集，论文对照采用对应数据集行。
- TCPT 的 Base 原文 Avg 与排除 ImageNet 后十数据集均值一致，引用时记录该汇总疑点。本轮 H、ECE、NLL 按固定规则自行计算。
- 正温度作用同一份 logits 的预测排序需要逐样本核对。论文中 TS 行与基线准确率差异的具体来源保留为复现核查问题。
- 温度网格、单模板参照、五轮 MaPLe 及验收后的工程修正均明确写入记录。

## 13 故障处理与下一步入口

| 现象 | 处理与验收 |
| --- | --- |
| 数据下载或 JSON 解析失败 | 核对来源与内容，通过浏览器下载上传后重新解析和校验 |
| 新配置项报不存在 | 完成第 5 节接口注册，再检查展开后的配置 |
| Base/New 标签越界或类别顺序不一致 | 检查分组、重映射、检查点 token 缓冲和 logits 列顺序 |
| 损失或梯度出现非有限值 | 保存运行日志和批次信息，检查精度与正则 FP32 统计，修正后重新验收受影响配置 |
| 温度拟合出现空验证集或候选类别不符 | 核对 Base、val、类别映射与验证清单 |
| 保存重载后预测不一致 | 核对评测模式、预处理、GPA 检查点和模型加载日志 |
| 超出时间预算 | 先完成 B 的实际运行、离线表图及证据汇总，按实际覆盖范围制作汇报 |

当前操作顺序为第 4 节数据权重准备、第 5 节工作副本接口接入和验收，再进入第 6、7 节正式运行。

## 14 固定来源

| 来源 | 固定提交或记录 | 用途 |
| --- | --- | --- |
| PromptSRC | `bb95c77b634d63488f2cad81ff4a72d53bdd06d5` | 共同数据、MaPLe、PromptSRC 与 GPA |
| TCPT | `729232f3be19c8b7d31b9933a55f172e96ac4ef4` | 间隔正则、文本矩匹配与参照模板 |
| Dassl.pytorch | `c61a1b570ac6333bd50fb5ae06aea59002fb20bb` | 数据加载、训练与配置框架 |
| DPC | 本地归档 `01e6b599d6515c38f4d723a8ea844ac4d1abfadd` | 扩展时的双提示与困难负类代码依据 |
| 云端环境 | 11 月 10 日保存的 `records/environment_check.json` | 硬件、依赖和 GPU 验收起点 |

正式运行记录实际工作副本和工程差异。DPC 扩展时若采用新的提交，另行固定提交与变更依据。

