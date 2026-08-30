"""Generate paper tables and figures from scored dialogues.

Usage:
  python scripts/04_analyze.py --scores results/main_run/scores.jsonl --out results/main_run/analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import run_full_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_full_analysis(args.scores, args.out)


if __name__ == "__main__":
    main()
