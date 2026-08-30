# SocraGuard

Reference implementation of *"SocraGuard: Benchmarking Adversarial Robustness of Socratic LLM Tutors and a Skill-based Defense"*.

> **Mechanism claim.** Multi-turn adversarial erosion is the primary failure mode of Socratic LLM tutors. Per-turn constraint re-injection is the only mechanism we identify that resists it — independent of model, prompt content, or framework.

## Repository structure

```
ICONIP2026_SocraGuard/
├── README.md                 — this file
├── REPRODUCE.md              — step-by-step reproduction of paper figures
├── requirements.txt
├── Dockerfile
├── .gitignore
├── configs/
│   ├── api_keys.example.yaml
│   ├── models_mvp.yaml         — MVP profile (DeepSeek + MiniMax + GLM)
│   ├── models_lncs.yaml        — LNCS profile (Claude + GPT-4o + DeepSeek + Qwen)
│   ├── models.yaml             — fallback when profile-specific not found
│   └── attack_templates.yaml   — 8 attack classes × bilingual templates
├── data/
│   └── socraguard_skill.md     — the SKILL specification (Appendix C)
├── src/
│   ├── llm_client.py           — unified OpenAI-compatible client + concurrency
│   ├── attacks.py              — 8-class taxonomy + 10-turn erosion ladder
│   ├── defenses.py             — D0..D4 (D2-vs-D3 same-content contrast)
│   ├── skill.py
│   ├── seeds.py                — Math/Code/Science/Chinese seed loaders
│   ├── benchmark.py            — JSONL I/O
│   ├── experiment.py           — main loop, concurrency, resume
│   ├── judge.py                — 3-judge cross-source voting
│   ├── metrics.py              — ALR/PLR/ASR/Crash Turn
│   ├── stats.py                — Bootstrap CI / Fisher / McNemar / Kappa
│   ├── analysis.py             — Tables + figures (publication-quality PDFs)
│   └── validator.py            — D4 validator (Qwen-1.5B + LoRA)
├── scripts/
│   ├── 00_check_apis.py
│   ├── 01_build_benchmark.py
│   ├── 02_run_experiment.py
│   ├── 03_run_judges.py
│   └── 04_analyze.py
├── tests/
│   ├── test_smoke.py
│   ├── test_defenses.py
│   └── test_stats.py
└── results/
```

## Two execution profiles

| Profile | Tutors | Judges | Cost | Time (8-way concurrent) |
|---------|--------|--------|------|------------------------|
| **MVP** | DeepSeek-V3, MiniMax-M2.7 | DeepSeek, MiniMax, GLM-4-Flash | ~¥60 | ~1.5 hour |
| **LNCS** | Claude 3.5 Sonnet, GPT-4o-mini, DeepSeek-V3, Qwen2.5-72B | Claude Sonnet, GPT-4o, DeepSeek-V3 | ~¥2400 | ~10 hours |

Switch profiles with the `--profile` flag or `SOCRAGUARD_PROFILE` env var.

## Quick start (MVP, 30 min start to finish)

```bash
git clone <repo-url> && cd ICONIP2026_SocraGuard
pip install -r requirements.txt
cp configs/api_keys.example.yaml configs/api_keys.yaml
# Edit api_keys.yaml: fill DeepSeek + MiniMax + Zhipu keys for MVP

python scripts/00_check_apis.py --backends deepseek minimax glm_flash

python scripts/01_build_benchmark.py --subject math --n_problems 5 \
    --output data/smoke.jsonl

python scripts/02_run_experiment.py --profile mvp \
    --tutors deepseek minimax \
    --defenses D0 D1 D2 D3 \
    --benchmark data/smoke.jsonl \
    --output results/smoke/dialogues.jsonl \
    --workers 4

python scripts/03_run_judges.py --profile mvp \
    --dialogues results/smoke/dialogues.jsonl \
    --output results/smoke/scores.jsonl \
    --workers 4

python scripts/04_analyze.py \
    --scores results/smoke/scores.jsonl \
    --out results/smoke/analysis
```

After this you have, in `results/smoke/analysis/`:

- `table_main.csv` — Table 2 of the paper
- `table_main_with_ci.csv` — same with bootstrap 95% CI
- `table_significance.csv` — Fisher + McNemar p-values
- `figure_heatmap.pdf` — Figure 3
- `figure_km_curve.pdf` — Figure 4
- `figure_cross_backend.pdf` — Figure 5
- `summary.json` — all numbers, machine-readable

## Full paper run (LNCS)

See `REPRODUCE.md`.

```bash
python scripts/01_build_benchmark.py --all-subjects --n_problems 100 \
    --output data/benchmark_full.jsonl

python scripts/02_run_experiment.py --profile lncs \
    --tutors claude_sonnet gpt4o_mini deepseek qwen_72b \
    --defenses D0 D1 D2 D3 \
    --benchmark data/benchmark_full.jsonl \
    --output results/lncs/dialogues.jsonl \
    --workers 8

python scripts/03_run_judges.py --profile lncs \
    --dialogues results/lncs/dialogues.jsonl \
    --output results/lncs/scores.jsonl \
    --workers 8

python scripts/04_analyze.py \
    --scores results/lncs/scores.jsonl \
    --out results/lncs/analysis
```

## D4 validator (optional)

```bash
python -m src.validator prepare \
    --scores    results/lncs/scores.jsonl \
    --dialogues results/lncs/dialogues.jsonl \
    --output    results/validator/train.jsonl

pip install torch transformers peft accelerate
python -m src.validator train \
    --train  results/validator/train.jsonl \
    --output results/validator/checkpoints/

python scripts/02_run_experiment.py --profile lncs \
    --tutors claude_sonnet deepseek qwen_72b gpt4o_mini \
    --defenses D4 \
    --validator results/validator/checkpoints/ \
    --benchmark data/benchmark_full.jsonl \
    --output results/lncs/dialogues.jsonl \
    --workers 8
```

## Mapping to the paper

| Paper artifact | Generated by |
|----------------|--------------|
| Table 1 (Attack taxonomy) | `configs/attack_templates.yaml` |
| Table 2 (Main results) | `04_analyze.py` → `table_main.csv` |
| Table 3 (Stage-wise erosion) | `04_analyze.py` → `table_stagewise.csv` |
| Table 5 (Human evaluation) | `src/stats.py::cohens_kappa` |
| Figure 3 (Heatmap) | `04_analyze.py` → `figure_heatmap.pdf` |
| Figure 4 (Kaplan-Meier) | `04_analyze.py` → `figure_km_curve.pdf` |
| Figure 5 (Cross-backend) | `04_analyze.py` → `figure_cross_backend.pdf` |
| Appendix B (Templates) | `configs/attack_templates.yaml` |
| Appendix C (SKILL.md) | `data/socraguard_skill.md` |

## Reproducibility

- Fixed seeds (default 0 / 42) for all random sampling
- `temperature=0.0` for all LLM calls
- Per-call model version logged
- Docker image provided (see `Dockerfile`)
- Resume on crash: re-run any script, finished items are skipped

## Tests

```bash
python tests/test_smoke.py
python tests/test_defenses.py
python tests/test_stats.py
```

## License

Apache-2.0.

## Citation

```bibtex
@inproceedings{socraguard2026,
  title = {SocraGuard: Benchmarking Adversarial Robustness of Socratic LLM Tutors and a Skill-based Defense},
  author = {Anonymous},
  booktitle = {Proceedings of ICONIP},
  year = {2026}
}
```
