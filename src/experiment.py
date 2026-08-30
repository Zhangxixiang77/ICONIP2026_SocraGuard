"""Main experiment loop.

For each (tutor, defense, scenario), run the multi-turn dialogue and
persist the full transcript. The persisted output (one line per
dialogue) is the authoritative experimental artifact — judges and
analyses read from it.

Concurrency model:
- Within a dialogue: turns are sequential (the tutor's response at
  turn t is required to construct turn t+1's history).
- Across dialogues: we can run many dialogues concurrently using
  ThreadPoolExecutor. This is the default with --workers > 1.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from .attacks import AttackScenario
from .defenses import Defense, D4_SkillWithValidator
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    turn_idx: int
    student: str
    tutor: str
    stage: str
    source_class: str | None
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None = None
    regenerated: bool = False
    validator_risk: dict | None = None


@dataclass
class DialogueRecord:
    dialogue_id: str
    tutor_backend: str
    tutor_model_reported: str
    defense: str
    scenario_id: str
    attack_class: str
    language: str
    problem: str
    ground_truth: str
    turns: list[TurnRecord] = field(default_factory=list)
    timestamp_started: float = 0.0
    timestamp_finished: float = 0.0


def run_dialogue(
    tutor: LLMClient,
    defense: Defense,
    scenario: AttackScenario,
    max_turns: int | None = None,
    max_regen_attempts: int = 2,
) -> DialogueRecord:
    """Run a single dialogue end-to-end."""
    dialogue_id = f"{tutor.backend_name}__{defense.defense_id}__{scenario.scenario_id}"
    record = DialogueRecord(
        dialogue_id=dialogue_id,
        tutor_backend=tutor.backend_name,
        tutor_model_reported=tutor.model_id,
        defense=defense.defense_id,
        scenario_id=scenario.scenario_id,
        attack_class=scenario.attack_class,
        language=scenario.language,
        problem=scenario.problem,
        ground_truth=scenario.ground_truth,
        timestamp_started=time.time(),
    )

    history: list[dict] = []
    n_turns = len(scenario.turns)
    if max_turns is not None:
        n_turns = min(n_turns, max_turns)

    for turn_idx in range(n_turns):
        student_turn = scenario.turns[turn_idx]
        student_msg = student_turn.text

        messages = defense.build_messages(history, student_msg, turn_idx)

        t0 = time.time()
        regenerated = False
        validator_risk = None

        try:
            resp = tutor.chat(messages)
            tutor_text = resp.text
            err = None
            ptok, ctok = resp.prompt_tokens, resp.completion_tokens
            if turn_idx == 0:
                record.tutor_model_reported = resp.model_reported

            # Post-validate (D4 only)
            for attempt in range(max_regen_attempts):
                vresult = defense.post_validate(history, student_msg, turn_idx, tutor_text)
                if vresult is None:
                    break
                regenerated = True
                validator_risk = vresult["reason"]
                hardened_resp = tutor.chat(vresult["hardened_messages"])
                tutor_text = hardened_resp.text
                ptok = (ptok or 0) + (hardened_resp.prompt_tokens or 0)
                ctok = (ctok or 0) + (hardened_resp.completion_tokens or 0)

        except Exception as e:
            tutor_text = ""
            err = f"{type(e).__name__}: {e}"
            ptok = ctok = None
            logger.warning("Dialogue %s turn %d failed: %s",
                           dialogue_id, turn_idx, err)

        latency = time.time() - t0

        record.turns.append(TurnRecord(
            turn_idx=turn_idx,
            student=student_msg,
            tutor=tutor_text,
            stage=student_turn.stage,
            source_class=student_turn.source_class,
            latency_seconds=latency,
            prompt_tokens=ptok,
            completion_tokens=ctok,
            error=err,
            regenerated=regenerated,
            validator_risk=validator_risk,
        ))

        # Update history
        history = [m for m in messages if m["role"] != "system"]
        history.append({"role": "assistant", "content": tutor_text})

        if err is not None:
            break

    record.timestamp_finished = time.time()
    return record


def run_experiment(
    tutors: dict[str, LLMClient],
    defenses: dict[str, Defense],
    scenarios: list[AttackScenario],
    output_path: Path | str,
    *,
    skip_existing: bool = True,
    max_turns: int | None = None,
    workers: int = 1,
) -> None:
    """Run all (tutor × defense × scenario) combinations.

    workers: 1 = strictly sequential; >1 = concurrent dialogues
             (each tutor's API rate limit determines a sane upper bound).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if skip_existing and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["dialogue_id"])
                except Exception:
                    continue
        logger.info("Resume mode: %d existing dialogues will be skipped.",
                    len(done_ids))

    combos = [
        (tn, t, dn, d, s)
        for tn, t in tutors.items()
        for dn, d in defenses.items()
        for s in scenarios
    ]
    todo = [c for c in combos if
            f"{c[0]}__{c[2]}__{c[4].scenario_id}" not in done_ids]
    logger.info("Total combinations: %d, to do: %d",
                len(combos), len(todo))

    # Open file for append
    out_f = open(output_path, "a", encoding="utf-8")

    def _run_and_write(combo):
        tn, tutor, dn, defense, scenario = combo
        try:
            rec = run_dialogue(tutor, defense, scenario, max_turns=max_turns)
            return rec
        except Exception as e:
            logger.exception("Failed combo %s/%s/%s: %s",
                             tn, dn, scenario.scenario_id, e)
            return None

    try:
        if workers <= 1:
            for combo in tqdm(todo, total=len(todo)):
                rec = _run_and_write(combo)
                if rec is not None:
                    out_f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    out_f.flush()
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_run_and_write, c) for c in todo]
                for fut in tqdm(as_completed(futures), total=len(futures)):
                    rec = fut.result()
                    if rec is not None:
                        out_f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                        out_f.flush()
    finally:
        out_f.close()


def read_dialogues(path: str | Path) -> list[DialogueRecord]:
    out: list[DialogueRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["turns"] = [TurnRecord(**t) for t in d["turns"]]
            out.append(DialogueRecord(**d))
    return out
