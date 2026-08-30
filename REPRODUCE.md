# Reproducing the paper

This document walks through reproducing every figure and table in the
SocraGuard paper, end to end. Total wall-clock time on a single
machine with 8-way concurrent API calls: **~14 hours** for the full
run, plus ~1 hour for D4 training (optional).

## 0. Prerequisites

- Python 3.10+
- API keys (for the LNCS profile):
  - **OpenRouter** (Claude 3.5 Sonnet, GPT-4o, GPT-4o-mini)
  - **DeepSeek** ([platform.deepseek.com](https://platform.deepseek.com))
  - **Aliyun DashScope** (Qwen2.5-72B)
- ~¥2400 budget for API calls
- Optional: a single GPU with 16GB+ VRAM for D4 validator training

## 1. Install

```bash
git clone <repo-url> && cd ICONIP2026_SocraGuard
pip install -r requirements.txt
cp configs/api_keys.example.yaml configs/api_keys.yaml
# Fill in the API keys
```

Verify:
```bash
python scripts/00_check_apis.py \
    --backends claude_sonnet gpt4o_mini deepseek qwen_72b gpt4o_judge
```
Every backend should print "OK" within 3-5 seconds.

## 2. Build the benchmark

```bash
python scripts/01_build_benchmark.py \
    --all-subjects --n_problems 100 \
    --output data/benchmark_full.jsonl
```
Result: `~3,400 scenarios` (4 subjects × 100 problems × 8 attacks +
extra erosion scenarios).

For real datasets (MATH-500, MBPP, ScienceQA, CMMLU) instead of the
built-in fallbacks, add `--use-hf` (requires `pip install datasets`).

## 3. Run the main experiment

This is the longest step (~6-10 hours with workers=8):

```bash
python scripts/02_run_experiment.py --profile lncs \
    --tutors claude_sonnet gpt4o_mini deepseek qwen_72b \
    --defenses D0 D1 D2 D3 \
    --benchmark data/benchmark_full.jsonl \
    --output results/lncs/dialogues.jsonl \
    --workers 8
```

Crash-safe: if the run is interrupted, re-running the same command
resumes from where it left off (already-completed dialogues are skipped).

You can run a smaller pilot first by adding `--max_scenarios 200`.

## 4. Run the LLM judges

```bash
python scripts/03_run_judges.py --profile lncs \
    --dialogues results/lncs/dialogues.jsonl \
    --output results/lncs/scores.jsonl \
    --workers 8
```

Wall-clock: ~3-4 hours with workers=8.

## 5. Generate paper tables and figures

```bash
python scripts/04_analyze.py \
    --scores results/lncs/scores.jsonl \
    --out results/lncs/analysis
```

Output (`results/lncs/analysis/`):

- `table_main.csv` → paste into Table 2
- `table_main_with_ci.csv` → for the bootstrap-CI footnote
- `table_significance.csv` → for the "p<0.001" claim
- `table_stagewise.csv` → paste into Table 3
- `figure_heatmap.pdf` → Figure 3 in paper
- `figure_km_curve.pdf` → Figure 4 in paper
- `figure_cross_backend.pdf` → Figure 5 in paper
- `summary.json` → headline numbers

## 6. (Optional) Train and run the D4 validator

```bash
# 6a. Prepare training data
python -m src.validator prepare \
    --scores    results/lncs/scores.jsonl \
    --dialogues results/lncs/dialogues.jsonl \
    --output    results/validator/train.jsonl

# 6b. Train (GPU recommended)
pip install torch transformers peft accelerate
python -m src.validator train \
    --train  results/validator/train.jsonl \
    --output results/validator/checkpoints/

# 6c. Re-run only D4
python scripts/02_run_experiment.py --profile lncs \
    --tutors claude_sonnet gpt4o_mini deepseek qwen_72b \
    --defenses D4 \
    --validator results/validator/checkpoints/ \
    --benchmark data/benchmark_full.jsonl \
    --output results/lncs/dialogues.jsonl \
    --workers 8

# 6d. Re-judge new D4 dialogues
python scripts/03_run_judges.py --profile lncs \
    --dialogues results/lncs/dialogues.jsonl \
    --output results/lncs/scores.jsonl \
    --workers 8

# 6e. Re-generate analysis (now includes D4)
python scripts/04_analyze.py \
    --scores results/lncs/scores.jsonl \
    --out results/lncs/analysis
```

## 7. Human evaluation (optional but recommended)

The paper reports Cohen's kappa among 3 raters on 200 stratified samples.
We provide the rating template + Kappa computation:

```bash
# Produce 200 stratified samples for annotators
python -c "
import pandas as pd, json
df = pd.read_json('results/lncs/scores.jsonl', lines=True)
# Stratify by (defense, attack_class) and randomly pick 200
sample = df.groupby(['defense','attack_class']).sample(n=5, random_state=42)
sample.to_csv('results/lncs/human_eval/to_rate.csv', index=False)
"
# Distribute to raters; collect their JSON files; then:
python -c "
from src.stats import cohens_kappa
import pandas as pd
ra = pd.read_csv('results/lncs/human_eval/rater_a.csv')
rb = pd.read_csv('results/lncs/human_eval/rater_b.csv')
print(cohens_kappa(ra.leakage_level, rb.leakage_level, weighted='linear'))
"
```

## 8. Cost summary (LNCS profile, full run)

| API | Use | Cost (¥) |
|-----|-----|---------:|
| OpenRouter (Claude Sonnet tutor) | tutor | ~800 |
| OpenRouter (GPT-4o-mini tutor) | tutor | ~150 |
| OpenRouter (Claude Sonnet judge) | judge | (shared) |
| OpenRouter (GPT-4o judge) | judge | ~600 |
| DeepSeek (tutor + judge) | both | ~250 |
| Aliyun DashScope (Qwen-72B tutor) | tutor | ~150 |
| **Total** | | **~¥2400** |

Buffer suggested: +¥300 for retries/reruns → **¥2700**.

## 9. Time budget

| Step | Wall-clock (workers=8) |
|------|----------------------:|
| Step 2 (benchmark) | < 1 minute (no LLM) |
| Step 3 (main experiment) | ~6-10 hours |
| Step 4 (judges) | ~3-4 hours |
| Step 5 (analysis) | < 1 minute |
| Step 6 (D4, optional) | ~1 hour training + ~2 hours rerun |
| Step 7 (human eval) | depends on raters |

A standard schedule: kick off Step 3 in the evening, wake up to Step 4,
run Step 5 over coffee.

## 10. Pinned model versions (for camera-ready)

The `tutor_model_reported` field in each dialogue records the exact
model version returned by the API. Use this for the camera-ready
Reproducibility statement.

A typical run reports:

- `anthropic/claude-3.5-sonnet-20241022`
- `openai/gpt-4o-mini-2024-07-18`
- `openai/gpt-4o-2024-08-06`
- `deepseek-chat-v3.x`
- `qwen2.5-72b-instruct`
