"""Run LLM-as-judge on all dialogues.

Usage:
  python scripts/03_run_judges.py \
      --dialogues results/main_run/dialogues.jsonl \
      --output    results/main_run/scores.jsonl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge import score_all_dialogues


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="mvp", choices=["mvp", "lncs"])
    parser.add_argument("--judges", nargs="+", default=None,
                        help="Override the default judge panel from models config")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    import os
    os.environ["SOCRAGUARD_PROFILE"] = args.profile

    if args.restart:
        out = Path(args.output)
        if out.exists():
            out.unlink()

    score_all_dialogues(
        dialogues_path=args.dialogues,
        output_path=args.output,
        judge_backend_names=args.judges,
        skip_existing=True,
        workers=args.workers,
        profile=args.profile,
    )
    print(f"\nDone. Scores in {args.output}")


if __name__ == "__main__":
    main()
