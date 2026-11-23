# SSH 增强预检运行说明

本轮只生成运行入口。以下 SSH 和 GPU 命令留待用户确认、实例可用后执行。运行固定到 [同步报告](GITHUB_SYNC_REPORT.md) 中的代码 commit SHA，环境按 [预检清单](SSH_PREFLIGHT_CHECKLIST.md) 验收。

## 1. 登录并准备独立 checkout

在本地使用已配置的 SSH Host 或用户指定的连接信息登录租用实例。保留主机密钥验证。私钥、密码、访问令牌、云端环境密钥和 SSH 配置均留在各自的授权环境中。访问私有 GitHub 仓库使用远端既有的 GitHub 访问方式。

```powershell
ssh <现有服务器SSH别名>
```

远端使用流式 Git clone 和 sparse checkout 取得阶段 2C 文件，历史数据与源码走独立目录。将下列变量替换为确认值。实例已有项目 checkout 时可用独立 worktree 或独立 clone，始终以完整 commit SHA 为准。

```bash
set -euo pipefail
umask 077
BRANCH=feature/clip-enhancement-v1-ssh-pilot
COMMIT='<从 GITHUB_SYNC_REPORT.md 复制完整可执行 commit SHA>'
CHECKOUT='<持久盘上的新 Git checkout 绝对路径>'
HISTORY='/root/autodl-tmp/clip-course-v1-20261116'
REFERENCE="$HISTORY/outputs/course_robustness_v1/formal-v1-confirmed-01"
PYTHON='/root/autodl-tmp/clip-exp/env/bin/python'
RUN_ID='enhancement-v1-pilot-01'
TRANSFER='<持久盘上的日志目录绝对路径>'
mkdir -p "$TRANSFER"
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout --single-branch \
  --branch "$BRANCH" git@github.com:bityang15/CLIP-Cloud-Experiment.git "$CHECKOUT"
git -C "$CHECKOUT" sparse-checkout set --no-cone '/analysis/extension-v1/enhancement/'
git -C "$CHECKOUT" switch --detach "$COMMIT"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$COMMIT"
ENTRY="$CHECKOUT/analysis/extension-v1/enhancement/pilot.py"
```

## 2. 复用历史环境并执行前置检查

优先在同一租用账户恢复原持久盘。`HISTORY` 指向包含原 `course_eval`、`code`、`records`、`outputs/B`、`weights` 和 `datasets` 的运行树。`REFERENCE` 允许指向单独解包的正式结果根，目录下有 `run_manifest.json` 和 `predictions/`。Git checkout 只提供阶段 2C 代码。若使用历史 `test_dependencies`，仅在它确实存在、与正式环境一致时设置该路径。无需自启整个训练环境。

```bash
nvidia-smi
"$PYTHON" -B "$ENTRY" preflight --commit "$COMMIT" --history "$HISTORY" \
  --reference "$REFERENCE" --device cuda:0 > "$TRANSFER/$RUN_ID.preflight.json"
```

检查退出码和 JSON 的 `passed` 字段。发现数据、权重、包版本或 CUDA 缺项时停止，记录报告后按缺项单独恢复，再重跑 `preflight`。入口只检查和记录，环境安装需要另行确认。原 4090 环境的 PyTorch CUDA runtime 为 12.1、cuDNN 记录值为 90100。

原运行树缺失时，从已有离线备份恢复到新的历史运行树。`env/course_v1_release_ready.tar.gz` 含冻结的 `course_eval`，其中 11 个正式记录的源码/协议成员 SHA-256 已匹配。`env/formal-v1-confirmed-01-backup01.tar.gz` 含正式参考输出，使用历史验收记录校验后再解包。共享 CLIP 权重、数据和 18 个提示学习检查点另从已存在持久盘或受控备份恢复。Windows 历史 `code` 树有 115 个源码文件仅行尾与正式 Linux 原件不同，远端检查仍要求正式原始字节。恢复时用原 Linux 树，或在独立恢复目录复制并按正式清单 SHA-256 验证 LF 版本。历史配置、检查点和正式结果保持原样。 [历史来源](SSH_PREFLIGHT_CHECKLIST.md)

## 3. 有界 GPU 预检

仅在用户确认与前置检查通过后运行下列命令。每次用独立 `RUN_ID`，失败时保留输出和失败日志。不要用放宽阈值继续执行。该脚本对每个数据集/方法/Base 或 New 的六个未增强对照做严格历史一致性复核，再执行对应的六个增强条件。预检输出记录 PyTorch CUDA 事件耗时与内存峰值，并给出全量规模粗估。

```bash
"$PYTHON" -B "$ENTRY" pilot --commit "$COMMIT" --history "$HISTORY" \
  --reference "$REFERENCE" --device cuda:0 --run-id "$RUN_ID" \
  2>&1 | tee "$TRANSFER/$RUN_ID.log"
RUN_DIR="$CHECKOUT/outputs/course_enhancement_v1/$RUN_ID"
sha256sum "$RUN_DIR/SHA256SUMS.json" "$TRANSFER/$RUN_ID.log" "$TRANSFER/$RUN_ID.preflight.json"
```

若 `pipefail` 使失败的 `pilot` 提前退出，单独执行最后两行检查失败归档。`parity.json` 提供失败条件的数值对照，`failure.log` 和 `completion.json` 记录失败状态。`budget.json` 的正式耗时将 100 槽 CUDA forward 按批数伸缩，将其余加载/传输时间按图像数伸缩。压缩输出体积和峰值显存均为预检估计。

## 4. 回传、验证和关闭实例

从已验证 SSH 会话记录远端清单和外部日志的 SHA-256，作为接收端独立对照。将完整 run 目录和两份外部日志复制到本地新目录。使用已经记录的 commit、清单摘要与日志摘要验证文件。失败归档可加 `--allow-failed` 检查传输完整性，不能作为成功的预检结果。

```powershell
$sshHost = '<现有服务器SSH别名>'
$runId = 'enhancement-v1-pilot-01'
$remoteRun = '<远端 RUN_DIR 的绝对路径>'
$remoteLogs = '<远端 TRANSFER 的绝对路径>'
$backup = Join-Path $PWD 'env/stage2c-backups'
New-Item -ItemType Directory -Force -Path $backup | Out-Null
scp -r "${sshHost}:$remoteRun" $backup
scp "${sshHost}:$remoteLogs/$runId.log" "${sshHost}:$remoteLogs/$runId.preflight.json" $backup
$receipt = Join-Path $backup $runId
$commit = '<确认的 40 位 Git commit SHA>'
$manifestHash = '<在 SSH 中记录的 SHA256SUMS.json 摘要>'
& .\env\course-v1\Scripts\python.exe -B analysis/extension-v1/enhancement/pilot.py verify `
  --run-dir $receipt --commit $commit --manifest-sha256 $manifestHash
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $backup "$runId.log"),(Join-Path $backup "$runId.preflight.json")
```

将 commit、哈希、验收结果与本地备份位置存入单独的运行记录。完整正式实验需另获确认，沿用同样的回传和验收。正式结果完成且本地备份通过后，再由用户确认关闭实例。
