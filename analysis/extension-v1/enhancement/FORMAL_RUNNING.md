# 冻结增强 V1 的正式运行手册

正式入口运行两数据集、Base/New、三种退化 L2/L3。三个提示学习方法各执行 seed 1/2/3，Zero-shot CLIP 作为一份确定性预测执行 `seed1`。全量新增 240 份 Enhanced NPZ、249,300 行预测。配对对照来自已验收的 `formal-v1-confirmed-01`。正式运行沿用历史 100 张 batch 和原顺序的自然尾批。 [冻结协议](ENHANCEMENT_PROTOCOL_V1.md) · [正式对照锁](formal_inputs_v1.json)

阶段 4A 完成代码与运行前审查。使用 [阶段 4A 验收记录](FORMAL_READINESS.md) 中列出的正式代码 SHA。阶段 3 实测与成本推算保存在原报告和预算 JSON。 [GPU 预检报告](runs/enhancement-v1-pilot-20261124-01/STAGE3_PILOT_REPORT.md) · [资源预算](runs/enhancement-v1-pilot-20261124-01/RESOURCE_BUDGET.json)

## 1. 在远端检查固定 checkout

使用已验证的 SSH 连接。下列变量的路径和值从运行前验收记录取得。需要重新取代码时，采用公开 GitHub HTTPS sparse checkout。历史源码、数据和权重通过 `HISTORY` 单独读取。 [SSH 连接及历史资产位置](SSH_PILOT_RUNNING.md)

```bash
set -euo pipefail
umask 077
COMMIT='<正式运行代码的完整 SHA>'
CHECKOUT='<运行前已固定的独立远端 checkout>'
HISTORY='/root/autodl-tmp/clip-course-v1-20261116'
REFERENCE="$HISTORY/outputs/course_robustness_v1/formal-v1-confirmed-01"
PYTHON='/root/autodl-tmp/clip-exp/env/bin/python'
TRANSFER='<持久盘上的独立日志与备份目录>'
RUN_ID='enhancement-v1-formal-20261124-01'
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$COMMIT"
export PYTHONPATH="$HISTORY/test_dependencies${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OUTDATED_IGNORE=1 PYTHONUNBUFFERED=1
DIR="$CHECKOUT/analysis/extension-v1/enhancement"
mkdir -p "$TRANSFER"
"$PYTHON" -B "$DIR/formal.py" preflight --commit "$COMMIT" --history "$HISTORY" \
  --reference "$REFERENCE" --device cuda:0 > "$TRANSFER/$RUN_ID.readiness.json"
```

`preflight` 核对环境、全部 4,155 张原始测试图、共享权重、18 份提示权重的 SHA-256，以及 240 份 Corrupted 和 4 份 Clean 参考的 NPZ、metadata、有效配置及原候选类别顺序。它还检查磁盘能容纳运行结果、CPU 图像质量 CSV、日志和一份同机传输归档。通过条件是退出码 0 且 `passed=true`。本地备份盘在开始前也需有相应空间。

## 2. 获得确认后运行与续跑

首次运行使用新 `RUN_ID` 和新日志文件。正式入口执行一次完整前置检查，再通过历史 `restored_model` 和 `ExperimentMixin.test` 生成 Enhanced logits。旧推理和退化源码仍由历史目录提供。

```bash
test ! -e "$TRANSFER/$RUN_ID.attempt-01.log"
"$PYTHON" -B "$DIR/formal.py" run --commit "$COMMIT" --history "$HISTORY" \
  --reference "$REFERENCE" --device cuda:0 --run-id "$RUN_ID" \
  2>&1 | tee "$TRANSFER/$RUN_ID.attempt-01.log"
```

续跑在同一 checkout、历史根、参考根和相同 CUDA 设备上，用原 `RUN_ID` 加 `--resume`，日志另取 `attempt-02` 等名称。运行清单固定代码、算子、资产和运行时版本。入口在加载模型前重新核对所有已发布组合的文件和有效配置字节，并比对正式参考的标签、类别及源像素哈希。中断的 staging 目录挪入同一 run 的 `failed_staging/`，留下中断记录后重新计算未完成组合。已发布文件校验失败则停止，保留资料并从原备份恢复或另开 run。每个组合先写 staging，校验后原子发布。并行写入同一 run 会被系统文件锁拒绝。

## 3. 完成与 CPU 图像指标验收

成功的 GPU 运行包含 240 份预测、`metrics.csv`、`completion.json` 和 `SHA256SUMS.json`。每组都有预测、配置、检查点、参考哈希及耗时显存记录。增强 RGB 和归一化张量的独立签名必须覆盖 24,930 个图像条件，并在模型和 seed 间一致。进程退出码、`completion.json` 的 `passed`、240 个组合及本地重验共同作为正式验收条件。

完成 GPU 运行后，用同一历史图像资产执行独立 CPU 质量步骤。每张图只解码并裁一次，每种退化和强度各生成一次质量记录。以 Clean 裁图为参照，核查 Clean、Corrupted、Enhanced 像素 SHA-256 后计算 PSNR/SSIM，合计 24,930 行。该步骤只处理图片，可随原始数据放置位置安排 CPU 资源。零误差 PSNR 记 `inf`，相同的两个无穷值之差记 0，混合正负无穷的汇总差值留空并保留有限值计数。

```bash
RUN="$CHECKOUT/outputs/course_enhancement_v1_formal/$RUN_ID"
QUALITY="$CHECKOUT/outputs/course_enhancement_v1_quality/$RUN_ID"
RUN_HASH=$(sha256sum "$RUN/SHA256SUMS.json" | cut -d' ' -f1)
"$PYTHON" -B "$DIR/formal_analysis.py" quality --commit "$COMMIT" --history "$HISTORY" \
  --reference "$REFERENCE" --run-dir "$RUN" --manifest-sha256 "$RUN_HASH" --out "$QUALITY" \
  2>&1 | tee "$TRANSFER/$RUN_ID.quality.log"
sha256sum "$RUN/SHA256SUMS.json" "$QUALITY/quality_manifest.json"
```

## 4. SSH 备份和本地分析

记录远端 `SHA256SUMS.json`、`quality_manifest.json` 及传输归档的 SHA-256。打包完整 `RUN`、`QUALITY`、独立 `TRANSFER` 中该 run 的日志和前置报告，再通过 SSH 下载到本地专用备份目录。先核对压缩包哈希再解包。本地 `formal.py verify` 核验所有运行文件、完整组合矩阵、commit 和发布清单。`formal_analysis.py analyze` 验证独立质量表属于同一个已封存 run，再从 raw logits 复算统计。 [SSH 回传流程](SSH_PILOT_RUNNING.md)

```powershell
$clipPython = '.\env\course-v1\Scripts\python.exe'
$dir = 'analysis/extension-v1/enhancement'
$history = $PWD.Path
$reference = Join-Path $history 'outputs/course_robustness_v1/formal-v1-confirmed-01'
$commit = '<正式运行代码 SHA>'
$run = '<已解包的正式 run 目录>'
$quality = '<已解包的独立图像质量目录>'
$runHash = '<已在远端记录的 SHA256SUMS.json 摘要>'
$qualityHash = '<已在远端记录的 quality_manifest.json 摘要>'
& $clipPython -B "$dir/formal.py" verify --run-dir $run --commit $commit --manifest-sha256 $runHash
& $clipPython -B "$dir/formal_analysis.py" analyze --history $history --reference $reference `
  --commit $commit --run-dir $run --manifest-sha256 $runHash --quality-dir $quality `
  --quality-manifest-sha256 $qualityHash --out '<新建本地分析目录>'
```

分析输出包含两张配对指标表（逐 seed 及均值/样本标准差）、两张正确性转移表和一份独立图像质量汇总。差值统一为 Enhanced − Corrupted。H 在同一 seed 的 Base/New 准确率上计算后再跨 seed 汇总。New ECE 保留 10/15/20 bins，NLL 使用历史 FP64 计算。正确性转移同时记录四种状态以及 Top-1 类别变化。Zero-shot CLIP 的有效 seed 数为 1，样本标准差留空。GitHub 归档小型指标与报告，本地归档完整预测、日志和哈希。

## 预计资源与费用

阶段 3 对 240 份 Enhanced 的粗估为 **53.4 分钟、489.8 MiB 输出**，复用已验收的 Corrupted 对照。算式中 GPU 租用费约为 `0.89 × 实际每小时单价`，另计运行前校验、重试、CPU 图像指标与传输占用的实例时间。预检峰值保留显存为 1.527 GiB，正式峰值和耗时以本轮记录为准。预计占用按预检外推，大批次吞吐、数据读取与导出压缩均会影响实际值。 [预检资源预算](runs/enhancement-v1-pilot-20261124-01/RESOURCE_BUDGET.json)
