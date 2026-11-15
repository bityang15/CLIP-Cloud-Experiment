# 云服务器环境配置记录

配置日期为 2026 年 11 月 10 日，时区为 Asia/Shanghai。GPU 检查完成时间为 13 时 46 分。

## 连接与资源

SSH 地址为 `connect.cqa1.seetacloud.com`，端口为 `38532`，用户为 `root`。

首次连接采用首次信任方式，将 ED25519 主机指纹保存至本地 `实验复现/.ssh_known_hosts`。指纹为 `SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc`。

| 项目 | 检查结果 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA GeForce RTX 4090 D |
| GPU 显存 | 24564 MiB，约 24 GiB |
| NVIDIA 驱动 | 595.71.05 |
| 容器可用 CPU | 18 核 |
| 容器内存上限 | 64424509440 字节，即 60 GiB |
| 数据盘 | `/root/autodl-tmp`，50 GB |
| 配置完成时数据盘使用量 | 约 6.1 GB，剩余约 44 GB |

内存额度根据 cgroup 限制读取。`free -h` 显示的主机内存总量约为 755 GiB，容器实际额度采用上表数值。

## 执行操作

1. 使用 SSH 登录，检查系统、GPU 驱动、CPU、容器内存限制、磁盘、Python、Conda 和预装包。初始镜像为 Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128。初始镜像的 GPU 矩阵计算通过。
2. 在数据盘创建 `/root/autodl-tmp/clip-exp/`，建立 `code`、`datasets`、`outputs`、`weights`、`records`、`scripts`、`env` 和缓存目录。
3. 下载 Dassl.pytorch，并固定到提交 `c61a1b570ac6333bd50fb5ae06aea59002fb20bb`。
4. 从本地已归档代码上传 PromptSRC 和 TCPT。PromptSRC 对应提交 `bb95c77b634d63488f2cad81ff4a72d53bdd06d5`，TCPT 对应提交 `729232f3be19c8b7d31b9933a55f172e96ac4ef4`。两份源码合计 261 个文件，上传压缩包 SHA256 以及解压后逐文件 SHA256 检查均通过。
5. 在 `/root/autodl-tmp/clip-exp/env` 创建独立的 Python 3.10 Conda 环境。环境目录约占 5.6 GB。
6. 安装固定版本的 PyTorch、torchvision、NumPy、SciPy、scikit-learn、Pandas、Matplotlib、Pillow、ftfy，以及 Dassl 和两份代码需要的依赖。
7. 将 Dassl 的 `requirements.txt` 中一项 `tb-nightly` 依赖替换为 `tensorboard==2.18.0`。修改保存于 `records/dassl_dependency.patch`。
8. 执行 `pip check`，保存完整 `pip freeze` 和 Conda 显式包清单。
9. 在两份代码各自目录运行 `python train.py --help`，验证训练入口及模块导入。
10. 执行合成输入的 FP16 GPU 前向、反向传播、SGD 参数更新，以及 torchvision CUDA NMS 检查。损失、梯度和更新后的参数均为有限值。
11. 再次核查两份上传源码的逐文件校验值，检查通过。

环境版本和 TensorBoard 依赖替换属于本次环境配置时作出的工程选择。

## 最终主要版本

| 软件 | 版本 |
| --- | --- |
| Python | 3.10.22 |
| PyTorch | 2.4.1 |
| PyTorch CUDA 运行库 | 12.1 |
| torchvision | 0.19.1 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| scikit-learn | 1.5.2 |
| Pandas | 2.2.3 |
| Matplotlib | 3.9.2 |
| Pillow | 10.4.0 |
| ftfy | 6.1.1 |
| TensorBoard | 2.18.0 |
| Dassl | 0.6.3 |
| WILDS | 1.2.2 |
| pip | 24.3.1 |
| setuptools | 69.5.1 |
| wheel | 0.45.1 |

## 执行中遇到的问题与处理

- 首次连接时，本地沙箱发生域名解析失败。通过经自动审批的网络执行环境成功连接。
- 首次 Conda 求解时，配置的镜像源缺少指定的 pip 24.3.1。调整为先创建 Python 环境，再使用 pip 安装固定版本的 pip、setuptools 和 wheel。
- 初次使用官方 PyPI 下载时速度约为 35 KB/s。暂停该次下载后，切换至清华 PyPI 镜像，随后实测速度约为 13–18 MB/s。
- 国内镜像缺少 Dassl 要求的 `tb-nightly`。将这一项依赖改为稳定版 TensorBoard 2.18.0，重新安装后依赖一致性和训练入口检查通过。
- SSH 期间发生一次连接中断，随后重连。环境安装使用独立 screen 会话运行，并通过日志及退出码核查状态。

相关失败、重试和成功日志均保存在云端 `records` 目录。

## 检查范围与下一阶段

本次验证覆盖硬件可用性、包依赖一致性、合成输入 GPU 运算、FP16 梯度更新、CUDA 算子和源码训练入口导入。

下一阶段需要准备 DTD、Flowers102、官方划分文件和 CLIP ViT-B/16 权重，并完成实验计划要求的数据清单核查、三种配置的真实数据小批次预跑，以及 TCPT 与基线的统一训练入口核查。

## 进入实验环境

在 SSH 或 JupyterLab 终端执行下列命令。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/clip-exp/env
cd /root/autodl-tmp/clip-exp/code/PromptSRC
python train.py --help
```

## 文件位置

- `code/PromptSRC` 和 `code/TCPT` 保存上传的归档源码。
- `code/Dassl.pytorch` 保存固定提交及一处依赖修改。
- `scripts/bootstrap.sh` 保存环境依赖安装流程，运行前需要已有 Python Conda 环境与上述源码目录。
- `scripts/constraints.txt` 保存主要依赖版本约束。
- `scripts/verify_environment.py` 保存 GPU 检查程序。
- `records/environment_check.json` 保存检查结果。
- `records/pip_freeze.txt` 与 `records/conda_explicit.txt` 保存实际安装版本。
- `records/source_revisions.json` 和 `records/source_sha256_before.txt` 保存代码来源与校验值。
- `records/dassl_dependency.patch` 保存依赖修改。
- `records/bootstrap.log` 和 `records/gpu_verification.log` 保存最终成功日志。

云端以上相对路径均位于 `/root/autodl-tmp/clip-exp/`。
