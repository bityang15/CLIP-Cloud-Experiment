# 实验资料复用

## 浏览与核查

仓库首页链接到 B、E 报告、CSV 和 PNG/PDF 图。逐次运行保存在 `outputs/B`、`outputs/E`，其中 `config.yaml` 保存实际参数，`console.command.json` 保存执行命令，`run_status.json` 保存完成状态，`predictions` 保存预测 logits 与元数据，`temperature.json` 保存 Base 验证温度。

运行下面的命令，可以按原始 B、E 文件清单逐项检查大小和 SHA256，并检查正式工作副本。Python 标准库即可运行。

```bash
python tools/verify_archive.py
python tools/verify_archive.py --with-checkpoints
```

第二条命令需要将保存在本地的 30 个检查点放回清单所示路径。原始清单分别有 849、880 个文件，正式工作副本分别有 117、331 个文件。清单中重复出现的共享模块按原路径保留。默认核查会跳过检查点。[本地检查点清单](../records/checkpoints.json)

## 原始运行环境

原脚本将实验根目录固定为 `/root/autodl-tmp/clip-exp`，训练队列使用 Linux 的 `fcntl` 文件锁。重用这些入口时，在新的 Linux/CUDA 环境将仓库放到该位置。运行前按环境记录安装 Python 3.10、PyTorch 2.4.1、torchvision 0.19.1 与固定依赖。[实际环境](environment.md) · [版本约束](../scripts/constraints.txt) · [安装快照](../records/pip_freeze.txt)

```bash
cd /root/autodl-tmp/clip-exp
conda create --prefix ./env python=3.10 pip -y
conda activate /root/autodl-tmp/clip-exp/env
python -m pip install pip==24.3.1 setuptools==69.5.1 wheel==0.45.1
python -m pip install -c scripts/constraints.txt numpy pillow torch==2.4.1 torchvision==0.19.1
python -m pip install -c scripts/constraints.txt scipy scikit-learn pandas matplotlib
python -m pip install -c scripts/constraints.txt --no-build-isolation -e code/Dassl.pytorch
python -m pip install -c scripts/constraints.txt -r code/PromptSRC_expB/requirements.txt -r code/DPC_expE/requirements.txt
python -m pip check
python scripts/verify_environment.py
```

`code/Dassl.pytorch` 从实验记录的固定提交恢复，并应用原有 TensorBoard 依赖修正。E 使用它自身归档的 Dassl 工作副本，由 E 入口设置导入路径。`code/PromptSRC`、`code/TCPT` 保存最初上传云端的官方代码，261 个文件与环境配置时的哈希逐项一致。`scripts/bootstrap.sh` 保留最初配置环境的原始脚本。上面的命令使用本仓库正式工作副本目录。

## 数据与预训练权重

仓库保存 DTD、Flowers102 的官方划分与类别名、六组逐种子训练/验证/测试清单，以及 SPLE 官方标注和受限负样本库。图像与 CLIP 预训练权重可通过归档的数据准备脚本恢复。

```bash
bash scripts/prepare_data.sh
python scripts/audit_data.py
```

预训练权重为 CLIP ViT-B/16，SHA256 为 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`。数据准备脚本会校验下载文件，已有官方划分可以直接复用。[数据来源](../datasets/split_provenance.json) · [原始数据验收](../records/data_checks.json) · [图像与清单哈希](../records/data_manifest.json)

## 原有运行入口

下列入口使用归档实现与固定配方。B 队列会复用通过校验的同配方已完成运行，E 队列会创建新的 attempt。运行状态、训练日志与结果表会随执行更新。需要保留本次备份时，在独立副本中运行。

```bash
cd /root/autodl-tmp/clip-exp/code/PromptSRC_expB
python preflight.py
python run_matrix.py
python analyze_results.py

cd /root/autodl-tmp/clip-exp/code/DPC_expE
python preflight_e.py
python run_matrix_e.py
python /root/autodl-tmp/clip-exp/scripts/analyze_e.py
```

重新生成分析时，可分别运行 `analyze_results.py` 和 `scripts/analyze_e.py`。这些脚本读取既有预测，并更新温度、CSV、图和分析验收记录。

## 整理记录

2026 年 11 月 15 日的整理依据为本地已通过哈希校验的云端备份。B、E 的代码、预测、日志和正式检查点按原始清单重新核对。报告中的 Windows 绝对链接转换为仓库相对链接，B 协议中的前期备选路线引用已从整理副本移除。

原有证据压缩包中的代码、日志、预测和图表以可浏览目录保存，压缩包原始哈希仍见备份验收记录。30 个检查点按用户选择保留在本地，仓库保存其路径、大小与 SHA256。新增的 Dassl 依赖通过固定提交恢复，其余正式工作副本保留原有逐文件哈希。[B 本地备份验收](../records/B_local_backup_checks.json) · [E 本地备份验收](../records/E_local_backup_checks.json) · [本次仓库核查](../records/repository_checks.json)
