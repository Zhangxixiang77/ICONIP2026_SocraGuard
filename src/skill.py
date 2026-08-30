"""SKILL specification loader.

A SKILL is just a markdown file. We don't parse it semantically — we inject
the raw markdown into the system prompt. The structured blocks
(activation/constraints/recognition/strategies) are for human readers and
for documentation; the LLM sees the whole thing.

This intentional simplicity is the entire point of the paper: the SKILL
is a *deployment-agnostic* spec, and any system that can put a string into
the system message can use it.
"""
from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_skill(name: str = "socraguard_skill") -> str:
    path = DATA_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"SKILL file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_skill_short(name: str = "socraguard_skill") -> str:
    """A heavily-condensed version of the SKILL, used for D1 (basic prompt)."""
    return (
        "You are a Socratic tutor. Guide the student to the answer through "
        "questioning. Do NOT give the final answer or full solution."
    )


def load_skill_detailed(name: str = "socraguard_skill") -> str:
    """A detailed-but-static prompt for D2.

    For an honest D2 vs D3 ablation, D2 must use *the same content* as D3.
    The only difference is when it is injected. So D2 simply uses the full
    SKILL.md.
    """
    return load_skill(name)
