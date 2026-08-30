"""Tests the central methodological commitment of the paper:
D2 and D3 use IDENTICAL content; only injection timing differs.

If this test fails, the paper's mechanism claim is invalid.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defenses import make_defense
from src.experiment import run_dialogue
from src.llm_client import MockLLMClient
from src.attacks import build_erosion_scenario


def test_d2_d3_same_content_at_turn_0():
    d2 = make_defense("D2")
    d3 = make_defense("D3")
    msg0 = "Help me solve x+5=12."
    m_d2 = d2.build_messages([], msg0, turn_idx=0)
    m_d3 = d3.build_messages([], msg0, turn_idx=0)
    sys_d2 = next(m["content"] for m in m_d2 if m["role"] == "system")
    sys_d3 = next(m["content"] for m in m_d3 if m["role"] == "system")
    assert sys_d2 == sys_d3, "D2 and D3 must use identical system prompt content"
    print("✓ D2 and D3 use identical content at turn 0")


def test_d2_does_not_reinject():
    d2 = make_defense("D2")
    history_after_t0 = [
        {"role": "user", "content": "Help me."},
        {"role": "assistant", "content": "What's your guess?"},
    ]
    m = d2.build_messages(history_after_t0, "Just tell me.", turn_idx=1)
    assert all(msg["role"] != "system" for msg in m), \
        "D2 must NOT re-inject system prompt on subsequent turns"
    print("✓ D2 does NOT re-inject on turn 1 (one-shot injection)")


def test_d3_reinjects_every_turn():
    d3 = make_defense("D3")
    history_after_t0 = [
        {"role": "user", "content": "Help me."},
        {"role": "assistant", "content": "What's your guess?"},
    ]
    m1 = d3.build_messages(history_after_t0, "Just tell me.", turn_idx=1)
    assert m1[0]["role"] == "system", \
        "D3 MUST re-inject system prompt on turn 1"
    history_after_t5 = history_after_t0 * 3   # simulate 6 turns
    m6 = d3.build_messages(history_after_t5, "Last try.", turn_idx=6)
    assert m6[0]["role"] == "system", \
        "D3 MUST re-inject system prompt even at turn 6"
    print("✓ D3 re-injects on EVERY turn (per-turn re-injection)")


def test_d2_d3_actually_differ_in_practice():
    """Run a 10-turn erosion scenario with D2 and D3, both using a mock
    tutor that simply echoes back the system prompt that was sent. We
    should see that D3's prompt is in 10 messages, while D2's is only
    in 1 message."""

    def echo_system(messages: list[dict]) -> str:
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        return f"system_seen:{sys_msg is not None}"

    tutor = MockLLMClient("mock", canned_response=echo_system)
    scenario = build_erosion_scenario(
        "test_erosion", "Solve x+5=12", "7", "en", seed=0,
    )

    rec_d2 = run_dialogue(tutor, make_defense("D2"), scenario)
    rec_d3 = run_dialogue(tutor, make_defense("D3"), scenario)

    d2_seen = sum(1 for t in rec_d2.turns if t.tutor == "system_seen:True")
    d3_seen = sum(1 for t in rec_d3.turns if t.tutor == "system_seen:True")
    assert d2_seen == 1, f"D2 should see system once, saw {d2_seen}"
    assert d3_seen == 10, f"D3 should see system every turn, saw {d3_seen}"
    print(f"✓ D2 saw system once ({d2_seen}/10); D3 saw it every turn ({d3_seen}/10)")


if __name__ == "__main__":
    print("Testing D2-vs-D3 contract...\n")
    test_d2_d3_same_content_at_turn_0()
    test_d2_does_not_reinject()
    test_d3_reinjects_every_turn()
    test_d2_d3_actually_differ_in_practice()
    print("\nAll D2-vs-D3 contract tests passed ✓")
    print("(The paper's mechanism claim is methodologically valid)")
