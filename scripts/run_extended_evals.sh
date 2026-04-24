#!/usr/bin/env bash
# Driver: run extended-eval suite on all 7 LM checkpoints, split across GPU 0 and GPU 1.
# Uses the peer worktrees that carry the full training configs + mamba2 src changes.
# Results land in /scratch/.../extended-evals/results/raw/*.json.
set -euo pipefail

OUT_ROOT="/scratch/gpfs/EHAZAN/tharuntk/worktrees/extended-evals/results/raw"
LOG_ROOT="/scratch/gpfs/EHAZAN/tharuntk/worktrees/extended-evals/logs"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

WT_50M="/scratch/gpfs/EHAZAN/tharuntk/worktrees/lm-50m-sweep"
WT_150M="/scratch/gpfs/EHAZAN/tharuntk/worktrees/lm-150m-mamba2"

# Priority ICL tasks + rerun piqa/hellaswag as cross-check.
TASKS="piqa,hellaswag,winogrande,openbook_qa,sciq,arc_easy,commonsense_qa,social_iqa"
# Stretch perplexity-based (cheap) evals
TASKS_STRETCH="${TASKS},arc_easy_ppl,trivia_qa_wiki_ppl,natural_qs_open_ppl"

run_one() {
  local gpu=$1
  local wt=$2
  local cfg=$3
  local ckpt=$4
  local name=$5
  local tasks=$6
  local logf="${LOG_ROOT}/eval-${name}.log"
  local outf="${OUT_ROOT}/${name}.json"
  echo "[launch gpu=${gpu}] ${name} -> ${outf}"
  (
    cd "$wt"
    CUDA_VISIBLE_DEVICES="$gpu" \
      uv run python scripts/eval_lm_extended.py \
      "$cfg" \
      "$ckpt" \
      --tasks "$tasks" \
      --out "$outf" \
      >"$logf" 2>&1
  )
}

# Stage 1: 150m attn (gpu 0) + 150m mamba2 (gpu 1)
run_one 0 "$WT_150M" "configs/lm/scale_150m/attn.yaml" \
  /scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-150m-attn/step2862/model.pt \
  "150m-attn" "$TASKS_STRETCH" &
PID0=$!
run_one 1 "$WT_150M" "configs/lm/scale_150m/mamba2.yaml" \
  /scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-150m-mamba2/step2862/model.pt \
  "150m-mamba2" "$TASKS_STRETCH" &
PID1=$!
wait $PID0 $PID1

# Stage 2: 150m alt-attn-mamba2 (gpu 0) + 50m attn (gpu 1)
run_one 0 "$WT_150M" "configs/lm/scale_150m/alt_attn_mamba2.yaml" \
  /scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-150m-alt-attn-mamba2/step2862/model.pt \
  "150m-alt-attn-mamba2" "$TASKS_STRETCH" &
PID0=$!
run_one 1 "$WT_50M" "configs/lm/scale_50m/attn.yaml" \
  /scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-50m-attn/step954/model.pt \
  "50m-attn" "$TASKS_STRETCH" &
PID1=$!
wait $PID0 $PID1

# Stage 3: 50m mamba2 (gpu 0) + 50m alt-attn-mamba2 (gpu 1)
run_one 0 "$WT_50M" "configs/lm/scale_50m/mamba2.yaml" \
  /scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-50m-mamba2/step954/model.pt \
  "50m-mamba2" "$TASKS_STRETCH" &
PID0=$!
run_one 1 "$WT_50M" "configs/lm/scale_50m/alt_attn_mamba2.yaml" \
  /scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-50m-alt-attn-mamba2/step954/model.pt \
  "50m-alt-attn-mamba2" "$TASKS_STRETCH" &
PID1=$!
wait $PID0 $PID1

# Stage 4: 50m headwise-mamba2 (gpu 0, solo)
run_one 0 "$WT_50M" "configs/lm/scale_50m/headwise_mamba2.yaml" \
  /scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-50m-headwise-mamba2/step954/model.pt \
  "50m-headwise-mamba2" "$TASKS_STRETCH" &
PID0=$!
wait $PID0

echo "All done."
