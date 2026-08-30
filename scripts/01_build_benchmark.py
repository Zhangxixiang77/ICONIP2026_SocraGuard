"""Build the SocraGuard-Bench benchmark file.

Subjects supported: math, code, science, chinese.

Usage examples:
  # Single subject, MVP-size
  python scripts/01_build_benchmark.py --subject math --n_problems 50

  # All four subjects, full LNCS-size run
  python scripts/01_build_benchmark.py --all-subjects --n_problems 100 \
      --output data/benchmark_full.jsonl

  # Use real HuggingFace datasets (MATH-500, MBPP, ScienceQA, CMMLU)
  python scripts/01_build_benchmark.py --all-subjects --use-hf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks import (
    SINGLE_TURN_CLASSES,
    build_erosion_scenario,
    build_single_turn_scenario,
)
from src.benchmark import write_benchmark
from src.seeds import load_seeds


def build_for_subject(
    subject: str,
    seeds: list[tuple[str, str]],
    *,
    language: str,
    base_seed: int,
) -> list:
    scenarios = []
    sid = 0
    for prob_idx, (problem, answer) in enumerate(seeds):
        for cls in SINGLE_TURN_CLASSES:
            sid += 1
            scenarios.append(build_single_turn_scenario(
                scenario_id=f"{subject}_{prob_idx:03d}_{cls}",
                problem=problem,
                ground_truth=answer,
                attack_class=cls,
                language=language,
                seed=base_seed + sid,
            ))
        # one erosion per problem
        sid += 1
        scenarios.append(build_erosion_scenario(
            scenario_id=f"{subject}_{prob_idx:03d}_EROSION",
            problem=problem,
            ground_truth=answer,
            language=language,
            seed=base_seed + sid,
        ))
    return scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=None,
                        choices=["math", "code", "science", "chinese"])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--n_problems", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--use-hf", action="store_true",
                        help="Use real HuggingFace datasets if available.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    subjects: list[str]
    if args.all_subjects:
        subjects = ["math", "code", "science", "chinese"]
    elif args.subject is not None:
        subjects = [args.subject]
    else:
        parser.error("Specify --subject SUBJECT or --all-subjects")

    if args.output is None:
        if len(subjects) == 1:
            args.output = f"data/benchmark_{subjects[0]}.jsonl"
        else:
            args.output = "data/benchmark_full.jsonl"

    LANG_MAP = {"math": "en", "code": "en", "science": "en", "chinese": "zh"}

    all_scenarios = []
    for subject in subjects:
        seeds = load_seeds(subject, args.n_problems,
                           use_huggingface=args.use_hf,
                           seed=args.seed)
        scenarios = build_for_subject(
            subject, seeds, language=LANG_MAP[subject],
            base_seed=args.seed * 10000 + len(all_scenarios),
        )
        all_scenarios.extend(scenarios)
        print(f"  {subject}: {len(seeds)} problems × 8 attacks = {len(scenarios)} scenarios")

    write_benchmark(all_scenarios, args.output)
    print(f"\nWrote {len(all_scenarios)} total scenarios to {args.output}")


if __name__ == "__main__":
    main()
