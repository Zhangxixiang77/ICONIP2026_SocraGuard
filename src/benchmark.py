"""Benchmark I/O: read/write attack scenarios as JSONL."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .attacks import AttackScenario, AttackTurn


def write_benchmark(scenarios: list[AttackScenario], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in scenarios:
            d = asdict(s)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def read_benchmark(path: str | Path) -> list[AttackScenario]:
    scenarios: list[AttackScenario] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            turns = [AttackTurn(**t) for t in d["turns"]]
            d["turns"] = turns
            scenarios.append(AttackScenario(**d))
    return scenarios


def iter_benchmark(path: str | Path) -> Iterator[AttackScenario]:
    """Memory-friendly iteration."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            turns = [AttackTurn(**t) for t in d["turns"]]
            d["turns"] = turns
            yield AttackScenario(**d)
