# 阶段 4A 正式运行前验收

2026-11-24，正式运行前验收通过。远端只执行 `formal.py preflight`，耗时 22 秒，退出码 0，`passed=true`。预检结束时正式输出目录尚未创建。 [正式运行手册](FORMAL_RUNNING.md)

## 代码对应关系

| 用途 | Git commit |
| --- | --- |
| 冻结增强与阶段 3 推理基准 | `23261793e4cfcab95ab82d4c23687c255c7d3c5a` |
| 阶段 3 GPU 验收及预算 | `750428396c581f34768bb0465020ce4fb5066c1d` |
| 本次验收的正式运行代码 | `504eeeeae1a9d357eabf4b726c0d459b182189df` |

正式代码位于 `feature/clip-enhancement-v1-ssh-pilot`。校验确认了冻结增强代码、参数和预检入口的字节一致性，历史 V1.0 推理与退化源码符合正式运行的源文件哈希。GitHub 已核对目标 SHA。远端 HTTPS sparse clone 超时，最终通过 SSH 传输哈希核验过的稀疏 Git 对象包，检出同一 commit 并检查正式入口、配置和输入锁的文件字节。冻结历史运行树继续独立提供资产。 [正式代码提交](https://github.com/bityang15/CLIP-Cloud-Experiment/commit/504eeeeae1a9d357eabf4b726c0d459b182189df)

## 验收证据

| 项目 | 已确认结果 |
| --- | --- |
| 正式矩阵 | 40 个模型/数据集/seed/Base–New 块，共 240 个 Enhanced 组合及 249,300 行预测 |
| 配对参考 | 240 份 Corrupted、4 份独立 Clean，NPZ、metadata、有效配置的 SHA-256 和样本/类别顺序均通过 |
| 远端原图 | DTD 1,692 张、Flowers102 2,463 张，全部原始图像 SHA-256 匹配 |
| 权重 | 全部 18 份提示学习检查点与共享 ViT-B/16 权重通过 SHA-256 验证 |
| 环境 | RTX 4090 D，驱动 595.71.05，Python 3.10.22，CUDA 12.1，cuDNN 90100，OpenCV 4.11.0 |
| 可用空间 | 远端正式输出卷 23.155 GiB，本地备份卷 357.053 GiB，均超过运行与归档的 1.105 GiB 预留量 |
| 本地测试 | 145 项通过，覆盖历史退化、阶段 3、正式划分、100/100/3 自然尾批、断点恢复、并发写锁、预测与配置篡改拒绝，以及配对指标和图像计数 |

前置检查原始 JSON 已通过 SSH 回传并核对远端 SHA-256，保存在 `env/stage4a-readiness/formal-readiness.json`，摘要为 `d4c50bca43add7141fa1a29dcc25332ab5e7222e0071df53aea8b6225394fd02`。正式输入锁摘要为 `6ee5082231cd8f92adf3ef5435c357a0256bee9a37198ca1fc9a9eac7686b63f`。 [对照锁](formal_inputs_v1.json) · [本地正式测试](test_formal_enhancement.py)

## 确认后执行的正式命令

下面的命令使用已验收的远端 checkout 和历史资产。首次运行应确认 run ID 和日志路径未使用，续跑保持相同参数并增加 `--resume`，更换日志文件名。 [续跑、验收和 SSH 备份步骤](FORMAL_RUNNING.md)

```bash
set -euo pipefail
HISTORY=/root/autodl-tmp/clip-course-v1-20261116
CHECKOUT=/root/autodl-tmp/clip-enhancement-v1-formal-20261124-01-verified
TRANSFER=/root/autodl-tmp/clip-enhancement-v1-formal-preflight-20261124-01
COMMIT=504eeeeae1a9d357eabf4b726c0d459b182189df
RUN_ID=enhancement-v1-formal-20261124-01
export PYTHONPATH="$HISTORY/test_dependencies${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OUTDATED_IGNORE=1 PYTHONUNBUFFERED=1
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$COMMIT"
test ! -e "$TRANSFER/$RUN_ID.attempt-01.log"
/root/autodl-tmp/clip-exp/env/bin/python -B "$CHECKOUT/analysis/extension-v1/enhancement/formal.py" run \
  --commit "$COMMIT" --history "$HISTORY" \
  --reference "$HISTORY/outputs/course_robustness_v1/formal-v1-confirmed-01" \
  --device cuda:0 --run-id "$RUN_ID" 2>&1 | tee "$TRANSFER/$RUN_ID.attempt-01.log"
```

## 成本与正式验收条件

Enhanced 推理沿用阶段 3 外推，预计 **53.4 分钟、489.8 MiB** 输出，租用费约为 `0.89 × 实际每小时单价`，另计校验、CPU 质量计算和传输。本地四张裁图上的 24 次串行图像质量计算加退化及增强处理用时约 0.78 秒。按此速率外推到 24,930 个条件约 13.4 分钟，未计原图解码、读取和 SHA-256，也未测远端 CPU 吞吐。 [阶段 3 资源预算](runs/enhancement-v1-pilot-20261124-01/RESOURCE_BUDGET.json)

正式验收要求 240 份 Enhanced raw logits 全部发布，249,300 行预测与正式参考严格配对，24,930 个增强输入在模型和 seed 间一致，`completion.json` 和 `SHA256SUMS.json` 可复核。独立 CPU 图像质量表必须全部覆盖对应条件并随日志与预测一起通过 SSH 备份和本地哈希验收，之后生成配对统计。 [正式运行手册](FORMAL_RUNNING.md)
