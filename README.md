# RoboTwin High-Level / Low-Level Handoff Report

Updated: 2026-08-07

本仓库记录 RoboTwin high-level / low-level handoff baseline、Gemma4-12B 相关实验，以及 `next / all` planner 调试结果。

## 1. 本周完成

本周主要做了三件事：

- 收口 high-level 输入 / 输出，删掉了未实际参与控制的字段。
- 增加并验证了 `next / all` 两类 planner 路径，以及 `same / switch` verifier 路径。
- 在 Gemma4-12B 上跑通了 forced-boundary、verifier 和 planner 相关测试。

## 2. 当前代码逻辑

### 2.1 Planner 模式

当前支持的 planner 模式包括：

- `inline_vlm`：VLM 直接输出下一条 subtask。
- `vlm_plan_all`：在 episode 开头一次性生成完整的 subtask 列表。
- `vlm_plan_next`：需要切换时才生成下一条 subtask，可配 prefetch。
- `profile_all` / `lazy_profile`：确定性调试模式。

### 2.2 Verifier / 输出

当前支持的高层输出形式包括：

- `minimal`
- `minimal_status`
- `completion_status`

其中 `completion_status` 用于 `same / switch` 风格判断；`minimal_status` 用于最小控制接口。

当前 verifier 的主输出是：

```json
{
  "action": "continue | switch",
  "subtask_status": "running | completed | failed | stuck | uncertain",
  "next_subtask": "",
  "evidence": "short visual reason"
}
```

说明：

- `confidence` 只做日志记录，不参与控制。
- `switch` 时清空旧 instruction 下的 action cache。
- `continue` 时保留当前 action cache。

### 2.3 Visual Context

当前 benchmark 的 boundary 判断使用单张当前 head-camera 图像。

- 不再依赖 keyframe 控制路径。
- 默认不做 history-based keyframe 选择。
- `next / all` 的默认逻辑不受 keyframe 影响。

### 2.4 `next` / `all` 逻辑

`vlm_plan_next`：

- 当前 subtask 发布给 low-level 后，立刻后台预取下一条 candidate subtask。
- 到 boundary 时，verifier 只判断当前 subtask 是否完成。
- 如果完成且 prefetch 已就绪，就直接发布预取结果。
- 如果没赶上，就走同步兜底生成。

`vlm_plan_all`：

- episode 开头一次性生成完整 subtask list。
- boundary 时只做 verifier 判断。
- 如果完成，就消费缓存里的下一条 subtask。
- 如果未完成，就保持当前 subtask 不变。

## 3. Gemma4-12B 结果

### 3.1 `place_burger_fries` 的 completion verifier

实验名：`gemma4_status_217_20260806`

- formal episodes: `n=10`
- `boundary_to_instruction_ready` mean: `632.4 ms`
- median: `631.9 ms`
- min / max: `617.5 / 648.4 ms`
- 说明：这是单张图 completion-verifier 的耗时，不含低层后续恢复执行。

### 3.2 `place_burger_fries` 的 `next / all` 对比

实验名：

- `gemma4_mtp_vlm_plan_next_prefetch_217_20260807`
- `gemma4_mtp_vlm_plan_all_217_20260807`
- `gemma4_mtp_vlm_plan_all_repeated_next_217_20260807`

结果：

- `vlm_plan_next_prefetch`: mean `424.3 ms`, median `422.7 ms`, min / max `410.8 / 450.8 ms`
- `vlm_plan_all`: mean `446.4 ms`, median `444.8 ms`, min / max `428.7 / 472.4 ms`
- `vlm_plan_all_repeated_next`: mean `441.6 ms`, median `431.7 ms`, min / max `424.4 / 470.0 ms`

结论：

- `next` 路径更适合做边界到下一条指令的预取。
- `all` 路径可用，但更依赖完整 subtask list 的稳定性。

### 3.3 六任务 `n=10` 结果

| RoboTwin task | Forced first-subtask end threshold | Actual trigger sim time / frame | Mean | Median | Std | Min / Max |
|---|---:|---|---:|---:|---:|---:|
| `place_burger_fries` | 2.17s | 2.20s / 55 | 779.5 ms | 786.7 ms | 90.5 ms | 653.8 / 906.2 ms |
| `place_cans_plasticbox` | 2.25s | 2.28s / 57 | 828.1 ms | 824.4 ms | 38.9 ms | 782.5 / 894.3 ms |
| `put_bottles_dustbin` | 4.01s | 4.04s / 101 | 1134.2 ms | 870.7 ms | 850.2 ms | 791.7 / 3551.4 ms |
| `stack_blocks_three` | 2.25s | 2.28s / 57 | 1072.2 ms | 1083.7 ms | 83.9 ms | 914.1 / 1190.8 ms |
| `stack_bowls_three` | 4.53s | 4.56s / 114 | 938.6 ms | 929.1 ms | 72.0 ms | 814.3 / 1037.3 ms |
| `place_dual_shoes` | 2.61s | 2.64s / 66 | 1041.2 ms | 1025.9 ms | 34.6 ms | 991.5 / 1099.5 ms |

Overall across the 6 tasks:

- mean: `965.6 ms`
- median: `899.3 ms`
- std: `361.5 ms`
- min / max: `653.8 / 3551.4 ms`

## 4. Action cache

当前 controller 会记录 high-level 切换时的 stale action cache 丢弃情况。

六任务 Gemma4 MTP 结果里：

- mean `stale_action_cache_discarded`: `4.33`
- median: `4.50`
- min / max: `1 / 6`
- distribution:
  - `1`: 10 samples
  - `4`: 20 samples
  - `5`: 10 samples
  - `6`: 20 samples

## 5. Demo 网页雏形

已经加了一个静态 demo browser：

- 页面：`docs/demo/index.html`
- 数据：`docs/demo/data/demos.json`
- 本地数据兜底：`docs/demo/data/demos.js`
- 视频目录：`docs/demo/assets/`

当前页面能展示：

- 每个 RoboTwin task 的 demo slot。
- 当前 subtask。
- verifier 的 `same / switch` 判断。
- 下一条 high-level instruction。
- handoff latency。
- action cache 行为。
- 六任务 timing summary。

网页已经附带 GitHub Pages 自动部署配置：

```text
.github/workflows/deploy-demo-pages.yml
```

仓库启用 GitHub Pages 后，公开网址为：

```text
https://233333nmc.github.io/highlow-handoff/
```

每次更新 `docs/demo/` 并推送到 `highlow-handoff-baseline` 或 `main` 后，GitHub Actions 会自动重新部署页面。视频文件和 demo 数据也会随仓库一起发布，因此后续只需要更新 `docs/demo/assets/`、`demos.json` 和 `demos.js`。

后续如果策略或 demo 视频更新，只需要：

1. 把视频放到 `docs/demo/assets/`。
2. 在 `docs/demo/data/demos.json` 和 `docs/demo/data/demos.js` 中填入对应 `video` 路径。
3. 更新 subtask、verifier output 和 timing 字段。

## 6. 复现

下面的命令都把缓存、日志、模型和评测结果写在 `high_level_memory` 个人目录下，不使用系统临时目录。

### 6.1 目录结构

建议使用以下目录布局：

```text
<WORK_ROOT>/
├── Hy-Embodied-0.5-VLA/       # 本仓库
├── RoboTwin/                   # RoboTwin simulator
├── checkpoints/                # low-level / high-level checkpoint
├── envs/                       # unified Python 3.10 evaluation environment
├── profile_results/            # handoff logs and timing results
├── model_cache/                # persistent Hugging Face cache
└── runtime_cache/              # Gemma runtime cache
```

本文中的 `<WORK_ROOT>` 指：

```text
/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
```

如果换到其他机器，只需要把下面命令中的 `WORK_ROOT` 改成对应的个人工作目录。

### 6.2 获取代码

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT"

git clone -b highlow-handoff-baseline \
  https://github.com/233333nmc/highlow-handoff.git \
  Hy-Embodied-0.5-VLA

git clone https://github.com/robotwin-Platform/RoboTwin.git RoboTwin
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"
```

如果仓库已经存在，不需要重复 clone：

```bash
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"
git checkout highlow-handoff-baseline
git pull
```

### 6.3 创建 RoboTwin + Hy-VLA 评测环境

评测使用统一的 Python 3.10 环境，使 RoboTwin simulator、Hy-VLA low-level policy 和 high-level handoff controller 在同一个进程中运行。

前置条件：

- Linux；
- 可用的 CUDA GPU；
- `/usr/bin/python3.10`；
- `uv`；
- Gemma4 MTP 复现实验需要两个当前租约可见的 GPU。

如果系统还没有 `uv`，先按本机环境安装。然后运行：

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

bash scripts/install_robotwin_hyvla_eval_env.sh
```

该脚本会创建：

```text
$WORK_ROOT/envs/robotwin_hyvla_eval/
```

并安装 CUDA PyTorch、Hy-VLA 推理依赖、RoboTwin simulator 依赖以及 cuRobo。脚本内部已经把 `TMPDIR`、Hugging Face cache、uv cache 和 matplotlib cache 指向个人目录。

### 6.4 下载 RoboTwin 仿真资源

当前实验是 RoboTwin inference/evaluation，不需要下载 RoboTwin training demonstrations。需要下载的是 simulator 运行所需的背景、物体和 embodiment 资源。

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

bash scripts/download_robotwin_assets_persistent.sh
```

资源会保存到：

```text
$WORK_ROOT/RoboTwin/assets/
```

### 6.5 下载 low-level checkpoint

Hy-VLA low-level checkpoint 来自 Hugging Face 的：

```text
tencent/Hy-Embodied-0.5-VLA-RoboTwin
```

推荐使用仓库脚本之外的持久 Hugging Face cache，并把模型保存到统一目录：

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
export HF_HOME="$WORK_ROOT/model_cache/huggingface"
export HF_HUB_CACHE="$WORK_ROOT/model_cache/huggingface/hub"
export TMPDIR="$WORK_ROOT/.cache/tmp"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TMPDIR"

hf download tencent/Hy-Embodied-0.5-VLA-RoboTwin \
  --local-dir "$WORK_ROOT/checkpoints/Hy-Embodied-0.5-VLA-RoboTwin"
```

最终目录应为：

```text
$WORK_ROOT/checkpoints/Hy-Embodied-0.5-VLA-RoboTwin/
```

### 6.6 下载 Gemma4-12B

Gemma4 主模型使用：

```text
google/gemma-4-12B-it
```

模型条款可能要求先在 Hugging Face 页面接受许可，并在当前 shell 中配置 `HF_TOKEN`。下载命令为：

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

# 如果 Hugging Face 需要认证：
# export HF_TOKEN=...

bash scripts/download_gemma4_12b.sh
```

主模型会保存到：

```text
$WORK_ROOT/checkpoints/google/gemma-4-12B-it/
```

Gemma4 native MTP 还需要一个兼容的 assistant checkpoint。该 checkpoint 不是由 `download_gemma4_12b.sh` 自动生成的，复现实验时需要单独准备，并通过下面的路径传给 server：

```text
$WORK_ROOT/checkpoints/google/gemma-4-12B-it-assistant/
```

如果不提供 `GEMMA4_ASSISTANT_MODEL_PATH`，可以启动普通 Gemma4 inference，但不能复现本文中的 native MTP 配置。

### 6.7 启动 Gemma4 API server

当前 high-level controller 通过本地 OpenAI-compatible HTTP API 调用 Gemma4。启动 server 的终端保持运行，然后在另一个终端运行 RoboTwin evaluation。

`GEMMA4_TRANSFORMERS_PATH` 必须指向支持 `gemma4_unified` 的 Transformers 版本目录。若该版本已经安装在脚本默认 cache 中，可以省略这一项；否则需要显式指定。

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

GEMMA4_PORT=8003 \
GEMMA4_MODEL_PATH="$WORK_ROOT/checkpoints/google/gemma-4-12B-it" \
GEMMA4_ASSISTANT_MODEL_PATH="$WORK_ROOT/checkpoints/google/gemma-4-12B-it-assistant" \
GEMMA4_DEVICE=cuda:1 \
GEMMA4_ASSISTANT_DEVICE=cuda:0 \
GEMMA4_NUM_ASSISTANT_TOKENS=4 \
GEMMA4_NUM_ASSISTANT_TOKENS_SCHEDULE=heuristic \
bash scripts/serve_gemma4_12b_transformers_api.sh
```

这里的 `cuda:0` 和 `cuda:1` 是当前租约可见 GPU 的相对编号，不要手动覆盖 `CUDA_VISIBLE_DEVICES`。如果当前租约只有一张 GPU，需要改用不带 assistant checkpoint 的普通推理配置，或者申请两张可见 GPU。

### 6.8 运行 `vlm_plan_next`

`vlm_plan_next` 的逻辑是：当前 instruction 发布后，立即预取下一条 candidate subtask；到 boundary 时，verifier 先判断当前 subtask 是否完成，如果完成且 candidate 已经 ready，就直接发布 candidate，否则同步生成作为兜底。

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

LOG_DIR="$WORK_ROOT/profile_results/robotwin_burger_fries/gemma4_vlm_plan_next_$(date +%Y%m%d_%H%M%S)" \
TEST_NUM=11 \
SEED=10000 \
TASK_NAME=place_burger_fries \
FORCED_BOUNDARY_S=2.17 \
MAX_FORCED_BOUNDARIES=1 \
STOP_AFTER_MEASUREMENT_HANDOFF=False \
HL_PLANNER_MODE=vlm_plan_next \
HL_PROMPT_MODE=completion_status \
HL_FORCE_VISUAL_ON_DECISION=True \
HL_RECENT_FRAMES_LENGTH=1 \
HL_VLM_MAX_IMAGES=1 \
GEMMA4_SERVER_URL=http://127.0.0.1:8003 \
bash scripts/eval_robotwin_burger_fries_forced_gemma4_12b_status.sh
```

### 6.9 运行 `vlm_plan_all`

`vlm_plan_all` 的逻辑是：episode 开头一次性生成完整 subtask list；之后在 boundary 处只做 completion verifier，确认完成后消费缓存中的下一条 subtask。

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

LOG_DIR="$WORK_ROOT/profile_results/robotwin_burger_fries/gemma4_vlm_plan_all_$(date +%Y%m%d_%H%M%S)" \
TEST_NUM=11 \
SEED=10000 \
TASK_NAME=place_burger_fries \
FORCED_BOUNDARY_S=2.17 \
MAX_FORCED_BOUNDARIES=1 \
STOP_AFTER_MEASUREMENT_HANDOFF=False \
HL_PLANNER_MODE=vlm_plan_all \
HL_PROMPT_MODE=completion_status \
HL_FORCE_VISUAL_ON_DECISION=True \
HL_RECENT_FRAMES_LENGTH=1 \
HL_VLM_MAX_IMAGES=1 \
GEMMA4_SERVER_URL=http://127.0.0.1:8003 \
bash scripts/eval_robotwin_burger_fries_forced_gemma4_12b_status.sh
```

两条命令使用相同的 RoboTwin task、2.17s forced boundary、单张当前 head-camera 图像和 completion verifier；区别只在于下一条 subtask 的规划时机。

### 6.10 结果和日志位置

每次评测会在 `LOG_DIR` 下写入：

```text
hierarchical.log
hierarchical_decisions.jsonl
```

RoboTwin 的 episode 结果会写到：

```text
$WORK_ROOT/RoboTwin/eval_result/
```

当前脚本默认运行 11 个 rollout，其中 episode 0 作为额外 runtime warm-up，正式 latency 统计使用 episode 1-10。

### 6.11 可选：多任务运行

如果需要复现六任务 Gemma4 结果，可以使用多任务脚本，并通过 `HL_PLANNER_MODE` 选择 `vlm_plan_next` 或 `vlm_plan_all`：

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

LOG_DIR="$WORK_ROOT/profile_results/robotwin_multitask_gemma4_12b_$(date +%Y%m%d_%H%M%S)" \
TEST_NUM=11 \
SEED=10000 \
HL_PLANNER_MODE=vlm_plan_next \
HL_PROMPT_MODE=completion_status \
HL_FORCE_VISUAL_ON_DECISION=True \
HL_RECENT_FRAMES_LENGTH=1 \
HL_VLM_MAX_IMAGES=1 \
GEMMA4_SERVER_URL=http://127.0.0.1:8003 \
bash scripts/eval_robotwin_multitask_gemma4_12b_status_n10.sh
```

## 7. 目前还在做的事

- 整理 action cache 统计结果表。
- 继续调 verifier completion criteria，减少 false negative。
- 继续调 prompt，让 planner 输出更合适粒度的 subtask。

## 8. 说明

- 当前 README 只保留我们这条 RoboTwin handoff 线的工作。
- 不再写 Hy-VLA 原论文式介绍和原始 benchmark 结果。
- 当前结果是 latency-only，不含 full-task success。
- 后续重点仍是 async pipeline、same/switch verifier 和 completion detection。
