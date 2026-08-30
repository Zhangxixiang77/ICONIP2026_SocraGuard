"""Run the main experiment: tutors × defenses × scenarios.

Examples:
  # MVP profile, 2 tutors × 4 defenses, sequential
  python scripts/02_run_experiment.py \
      --profile mvp --tutors deepseek minimax \
      --defenses D0 D1 D2 D3 \
      --benchmark data/benchmark_math.jsonl \
      --output results/mvp_run/dialogues.jsonl

  # LNCS profile, 4 tutors × 5 defenses, 8-way concurrent
  python scripts/02_run_experiment.py \
      --profile lncs \
      --tutors claude_sonnet gpt4o_mini deepseek qwen_72b \
      --defenses D0 D1 D2 D3 D4 \
      --validator results/validator/checkpoints/ \
      --benchmark data/benchmark_full.jsonl \
      --output results/lncs_run/dialogues.jsonl \
      --workers 8

  # Resume after a crash: just re-run, --skip-existing is on by default
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import read_benchmark
from src.defenses import make_defense, D4_SkillWithValidator
from src.experiment import run_experiment
from src.llm_client import build_clients


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="mvp", choices=["mvp", "lncs"])
    parser.add_argument("--tutors", nargs="+", required=True)
    parser.add_argument("--defenses", nargs="+", default=["D0", "D1", "D2", "D3"])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validator", default=None,
                        help="Path to validator checkpoint dir for D4")
    parser.add_argument("--max_scenarios", type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    os.environ["SOCRAGUARD_PROFILE"] = args.profile

    scenarios = read_benchmark(args.benchmark)
    if args.max_scenarios:
        scenarios = scenarios[:args.max_scenarios]
    print(f"Loaded {len(scenarios)} scenarios from {args.benchmark}")

    tutors = build_clients(args.tutors, profile=args.profile)
    print(f"Tutors: {list(tutors)}")

    # Load validator if D4 in the defenses
    validator_fn = None
    if "D4" in args.defenses and args.validator:
        try:
            from src.validator import Validator
            v = Validator.load(args.validator)
            validator_fn = v.predict
            print(f"Loaded validator from {args.validator}")
        except Exception as e:
            print(f"WARNING: failed to load validator ({e}). "
                  f"D4 will behave like D3.")

    defenses = {}
    for d in args.defenses:
        if d == "D4":
            defenses[d] = D4_SkillWithValidator(validator_fn=validator_fn)
        else:
            defenses[d] = make_defense(d)
    print(f"Defenses: {list(defenses)}")

    output = Path(args.output)
    if args.restart and output.exists():
        output.unlink()

    expected = len(tutors) * len(defenses) * len(scenarios)
    print(f"Expected total dialogues: {expected}")

    run_experiment(
        tutors=tutors,
        defenses=defenses,
        scenarios=scenarios,
        output_path=output,
        skip_existing=True,
        max_turns=args.max_turns,
        workers=args.workers,
    )

    print(f"\nDone. Dialogues written to {output}")


if __name__ == "__main__":
    main()
