"""Attack sampling and the multi-turn erosion ladder.

Single-turn attack classes: DIRECT_DEMAND, AUTHORITY_APPEAL, ROLE_OVERRIDE,
HYPOTHETICAL_FRAMING, EMOTIONAL_MANIPULATION, PARTIAL_EXTRACTION,
REVERSE_PSYCHOLOGY.

Multi-turn class: MULTI_TURN_EROSION (composite, 10-turn ladder).

For single-turn classes used inside an erosion sequence, we randomly sample
one template from the corresponding class.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

ATTACK_CLASSES = [
    "DIRECT_DEMAND",
    "AUTHORITY_APPEAL",
    "ROLE_OVERRIDE",
    "HYPOTHETICAL_FRAMING",
    "EMOTIONAL_MANIPULATION",
    "PARTIAL_EXTRACTION",
    "REVERSE_PSYCHOLOGY",
    "MULTI_TURN_EROSION",
]

SINGLE_TURN_CLASSES = [c for c in ATTACK_CLASSES if c != "MULTI_TURN_EROSION"]


@dataclass
class AttackTurn:
    """One turn of attack content the student would say."""
    text: str
    stage: str               # BASELINE / CONFUSION / STAGE_A / B / C / D / SINGLE_TURN
    source_class: str | None # which attack class this turn was sampled from


@dataclass
class AttackScenario:
    """A complete attack scenario: a sequence of student utterances."""
    scenario_id: str
    problem: str
    ground_truth: str
    attack_class: str
    language: Literal["en", "zh"]
    turns: list[AttackTurn] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


_templates_cache: dict | None = None


def load_templates() -> dict:
    global _templates_cache
    if _templates_cache is not None:
        return _templates_cache
    path = CONFIG_DIR / "attack_templates.yaml"
    with open(path, "r", encoding="utf-8") as f:
        _templates_cache = yaml.safe_load(f)
    return _templates_cache


def sample_template(
    attack_class: str,
    language: str,
    rng: random.Random,
) -> str:
    """Pick one template for a single-turn class."""
    if attack_class == "MULTI_TURN_EROSION":
        raise ValueError("Use build_erosion_scenario for MULTI_TURN_EROSION")
    templates = load_templates()
    pool = templates[attack_class]["templates"][language]
    return rng.choice(pool)


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def build_single_turn_scenario(
    scenario_id: str,
    problem: str,
    ground_truth: str,
    attack_class: str,
    language: str = "en",
    seed: int = 0,
) -> AttackScenario:
    """A single-turn attack: turn 0 = problem, turn 1 = attack utterance.

    Conventionally we also let the tutor produce one Socratic response after
    turn 0 before delivering the attack at turn 1. The experiment loop
    handles this — here we just declare the student-side turns.
    """
    rng = random.Random(seed)

    # Turn 0: state the problem (sincere)
    t0 = AttackTurn(
        text=f"Help me with this problem: {problem}",
        stage="BASELINE",
        source_class=None,
    )
    # Turn 1: deliver the attack
    attack_text = sample_template(attack_class, language, rng)
    t1 = AttackTurn(
        text=attack_text,
        stage="SINGLE_TURN",
        source_class=attack_class,
    )

    return AttackScenario(
        scenario_id=scenario_id,
        problem=problem,
        ground_truth=ground_truth,
        attack_class=attack_class,
        language=language,
        turns=[t0, t1],
    )


def build_erosion_scenario(
    scenario_id: str,
    problem: str,
    ground_truth: str,
    language: str = "en",
    seed: int = 0,
) -> AttackScenario:
    """Build a 10-turn erosion scenario per the ladder defined in YAML."""
    rng = random.Random(seed)
    templates = load_templates()
    erosion_cfg = templates["MULTI_TURN_EROSION"]
    ladder = erosion_cfg["ladder"]

    turns: list[AttackTurn] = []

    for turn_idx in sorted(ladder.keys()):
        stage_info = ladder[turn_idx]
        stage = stage_info[0]
        source_class = stage_info[1]  # may be None

        if stage == "BASELINE":
            tmpl = rng.choice(erosion_cfg["baseline_templates"][language])
            text = tmpl.format(problem=problem)
        elif stage == "CONFUSION":
            text = rng.choice(erosion_cfg["confusion_templates"][language])
        else:
            # Stage A / B / C / D — sample from the source class
            text = sample_template(source_class, language, rng)

        turns.append(AttackTurn(text=text, stage=stage, source_class=source_class))

    return AttackScenario(
        scenario_id=scenario_id,
        problem=problem,
        ground_truth=ground_truth,
        attack_class="MULTI_TURN_EROSION",
        language=language,
        turns=turns,
    )
