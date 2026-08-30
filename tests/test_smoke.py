"""End-to-end smoke test with a mock LLM (no API calls).

Run: python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks import (
    SINGLE_TURN_CLASSES,
    build_erosion_scenario,
    build_single_turn_scenario,
)
from src.defenses import make_defense
from src.experiment import run_dialogue
from src.llm_client import MockLLMClient
from src.seeds import load_seeds, MATH_FALLBACK


def test_single_turn_scenario():
    s = build_single_turn_scenario(
        "test_001", "What is 2+2?", "4", "DIRECT_DEMAND", "en", seed=0,
    )
    assert s.attack_class == "DIRECT_DEMAND"
    assert len(s.turns) == 2
    assert s.turns[0].stage == "BASELINE"
    assert s.turns[1].stage == "SINGLE_TURN"
    print("✓ single-turn scenario builds correctly")


def test_erosion_scenario():
    s = build_erosion_scenario(
        "test_erosion", "Solve x+5=12", "7", "en", seed=0,
    )
    assert s.attack_class == "MULTI_TURN_EROSION"
    assert len(s.turns) == 10
    stages = [t.stage for t in s.turns]
    assert "STAGE_A" in stages
    assert "STAGE_D" in stages
    print("✓ erosion scenario builds 10 turns with all stages")


def test_seeds_loader():
    seeds = load_seeds("math", n_problems=5, use_huggingface=False, seed=42)
    assert len(seeds) == 5
    assert all(isinstance(s, tuple) and len(s) == 2 for s in seeds)
    seeds_2 = load_seeds("math", n_problems=5, use_huggingface=False, seed=42)
    assert seeds == seeds_2, "seed=42 should be deterministic"
    print(f"✓ seeds loader: math returns {len(seeds)} (problem, answer) pairs")


def test_pipeline_with_mock():
    tutor = MockLLMClient(
        "mock_tutor",
        canned_response="That's a great question. What's your first guess?",
    )
    defense = make_defense("D3")
    scenario = build_single_turn_scenario(
        "test_e2e", "Solve x+5=12", "7", "DIRECT_DEMAND", "en", 0,
    )
    record = run_dialogue(tutor, defense, scenario)
    assert record.dialogue_id == "mock_tutor__D3__test_e2e"
    assert len(record.turns) == 2
    assert all(t.tutor != "" for t in record.turns)
    print(f"✓ end-to-end mock dialogue ran ({len(record.turns)} turns)")


def test_defenses_for_all_levels():
    for level in ["D0", "D1", "D2", "D3", "D4"]:
        d = make_defense(level)
        msgs = d.build_messages([], "test", 0)
        assert isinstance(msgs, list) and len(msgs) >= 1
    print("✓ D0..D4 all instantiate and produce messages")


if __name__ == "__main__":
    print("Running SocraGuard smoke tests...\n")
    test_single_turn_scenario()
    test_erosion_scenario()
    test_seeds_loader()
    test_pipeline_with_mock()
    test_defenses_for_all_levels()
    print("\nAll smoke tests passed ✓")
