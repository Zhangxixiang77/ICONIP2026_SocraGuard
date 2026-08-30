"""Sanity check: verify each configured backend is reachable and responding.

Usage:
  python scripts/00_check_apis.py
  python scripts/00_check_apis.py --backends deepseek minimax
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `src` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import build_clients, load_models_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends", nargs="+", default=None,
        help="Subset of backends to test. Default: all configured.",
    )
    parser.add_argument(
        "--profile", default="mvp", choices=["mvp", "lncs"],
        help="Which models config to load.",
    )
    parser.add_argument("--n_calls", type=int, default=3)
    args = parser.parse_args()

    import os
    os.environ["SOCRAGUARD_PROFILE"] = args.profile

    cfg = load_models_config(profile=args.profile)
    if args.backends is None:
        args.backends = list(cfg["backends"].keys())

    print(f"Testing {len(args.backends)} backends, {args.n_calls} calls each.\n")

    clients = build_clients(args.backends, profile=args.profile)
    summary = []
    for name, client in clients.items():
        latencies = []
        errors = 0
        for i in range(args.n_calls):
            try:
                t0 = time.time()
                resp = client.chat([
                    {"role": "user", "content": "Say 'OK' and nothing else."}
                ], max_tokens=10)
                lat = time.time() - t0
                latencies.append(lat)
                print(f"  [{name}] call {i+1}: {lat:.2f}s | reply: {resp.text[:40]!r}")
            except Exception as e:
                errors += 1
                print(f"  [{name}] call {i+1}: ERROR {type(e).__name__}: {e}")
        ok_rate = (args.n_calls - errors) / args.n_calls
        avg_lat = sum(latencies) / len(latencies) if latencies else float("nan")
        summary.append((name, ok_rate, avg_lat, errors))
        print()

    print("=" * 60)
    print(f"{'Backend':<15} {'Success rate':>14} {'Avg latency':>14} {'Errors':>10}")
    print("=" * 60)
    for name, ok_rate, avg_lat, errors in summary:
        print(f"{name:<15} {ok_rate*100:>12.0f} % {avg_lat:>12.2f} s {errors:>10}")
    print()

    bad = [s for s in summary if s[1] < 1.0]
    if bad:
        print("⚠  Some backends had errors. Check API keys and quotas.")
        sys.exit(1)
    print("✅ All backends operational.")


if __name__ == "__main__":
    main()
