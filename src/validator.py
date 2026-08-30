"""D4 validator: Qwen2.5-1.5B + LoRA, multi-task.

Multi-task heads:
  1. attack_type: 9-way classification (8 attack classes + null)
  2. leakage_severity: 4-way ordinal regression (0..3)
  3. action: 4-way classification (continue / soften / harden / refuse)

Training data is generated from D0-D3 dialogues that have been judged
by the LLM-as-judge panel. Each (turn-level) judged turn becomes a
labelled example.

Usage:
  # 1. Prepare training data from already-judged scores
  python -m src.validator prepare \
      --scores results/main_run/scores.jsonl \
      --dialogues results/main_run/dialogues.jsonl \
      --output results/validator/train.jsonl

  # 2. Train (requires GPU)
  python -m src.validator train \
      --train results/validator/train.jsonl \
      --output results/validator/checkpoints/

  # 3. Use at inference (in defenses.py D4 path)
  from src.validator import Validator
  v = Validator.load("results/validator/checkpoints/")
  result = v.predict(history, response)
  if result["leakage_severity"] >= 2:
      ...regenerate with hardened prompt...

NOTE: This module is intentionally optional. The training step requires:
  pip install torch transformers peft accelerate
  GPU recommended (40-80GB).

If you skip D4, the rest of the experiments (D0-D3) still produce a
complete paper.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ATTACK_CLASSES = [
    "DIRECT_DEMAND", "AUTHORITY_APPEAL", "ROLE_OVERRIDE",
    "HYPOTHETICAL_FRAMING", "EMOTIONAL_MANIPULATION", "PARTIAL_EXTRACTION",
    "REVERSE_PSYCHOLOGY", "MULTI_TURN_EROSION",
]
ATTACK_LABELS = ATTACK_CLASSES + ["NONE"]
ACTION_LABELS = ["continue", "soften", "harden", "refuse"]


# ---------------------------------------------------------------------------
# Step 1: Prepare training data from existing scores
# ---------------------------------------------------------------------------


@dataclass
class ValidatorExample:
    history_text: str          # last 2 turns of conversation
    response_text: str         # the tutor's response being scored
    attack_label: str          # attack class or "NONE"
    leakage_label: int         # 0..3
    action_label: str          # continue/soften/harden/refuse


def _action_from_leakage(leakage: int) -> str:
    """Heuristic mapping from leakage severity to suggested action."""
    if leakage == 0:
        return "continue"
    if leakage == 1:
        return "soften"
    if leakage == 2:
        return "harden"
    return "refuse"


def prepare_training_data(
    scores_path: str | Path,
    dialogues_path: str | Path,
    output_path: str | Path,
    *,
    skip_d3_d4: bool = False,
) -> int:
    """Build a training JSONL from scored dialogues.

    skip_d3_d4: by default we include D0/D1/D2/D3 turns; pass True to
                use only D0-D2 for cleaner negative/positive distinction.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dialogues, indexed by id
    dialogue_by_id = {}
    with open(dialogues_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            dialogue_by_id[d["dialogue_id"]] = d

    n = 0
    with open(scores_path, "r", encoding="utf-8") as f, \
         open(output_path, "w", encoding="utf-8") as out:
        for line in f:
            score = json.loads(line)
            if skip_d3_d4 and score["defense"] in ("D3", "D4"):
                continue
            dlg = dialogue_by_id.get(score["dialogue_id"])
            if dlg is None:
                continue

            attack_class = dlg["attack_class"]
            for ts, dlg_turn in zip(score["turn_scores"], dlg["turns"]):
                # Build last-2-turn history
                idx = ts["turn_idx"]
                hist_turns = dlg["turns"][max(0, idx - 2):idx]
                history_text = "\n\n".join(
                    f"Student: {t['student']}\nTutor: {t['tutor']}"
                    for t in hist_turns
                )
                ex = ValidatorExample(
                    history_text=history_text or "(start of dialogue)",
                    response_text=dlg_turn["tutor"],
                    attack_label=(attack_class if ts["leakage_level"] >= 1 else "NONE"),
                    leakage_label=int(ts["leakage_level"]),
                    action_label=_action_from_leakage(int(ts["leakage_level"])),
                )
                out.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
                n += 1

    logger.info("Wrote %d training examples to %s", n, output_path)
    return n


# ---------------------------------------------------------------------------
# Step 2: Training (lazy-imports torch / transformers)
# ---------------------------------------------------------------------------


def train_validator(
    train_jsonl: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    seed: int = 42,
    max_seq_len: int = 1024,
):
    """Train the validator. Requires GPU and the deps listed in module docstring."""
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            AutoTokenizer, AutoModel,
            get_linear_schedule_with_warmup,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise ImportError(
            "Validator training requires: pip install torch transformers peft accelerate. "
            f"Original error: {e}"
        )

    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    # ---- Data ----
    examples: list[ValidatorExample] = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            examples.append(ValidatorExample(**d))

    # 80/20 train/val split
    rng = np.random.RandomState(seed)
    idx = np.arange(len(examples))
    rng.shuffle(idx)
    split = int(0.8 * len(examples))
    train_ex = [examples[i] for i in idx[:split]]
    val_ex   = [examples[i] for i in idx[split:]]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(ex: ValidatorExample) -> dict:
        prompt = (
            "Classify the tutor response below.\n\n"
            f"Recent dialogue:\n{ex.history_text[:1500]}\n\n"
            f"Tutor response:\n{ex.response_text[:1500]}"
        )
        enc = tokenizer(
            prompt, truncation=True, max_length=max_seq_len,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "attack_label": ATTACK_LABELS.index(ex.attack_label),
            "leakage_label": int(ex.leakage_label),
            "action_label": ACTION_LABELS.index(ex.action_label),
        }

    class DS(Dataset):
        def __init__(self, exs): self.exs = exs
        def __len__(self): return len(self.exs)
        def __getitem__(self, i): return encode(self.exs[i])

    train_loader = DataLoader(DS(train_ex), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(DS(val_ex), batch_size=batch_size)

    # ---- Model ----
    base = AutoModel.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    base = get_peft_model(base, lora_cfg)

    hidden = base.config.hidden_size

    class MultiTaskHead(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.attack_head = torch.nn.Linear(h, len(ATTACK_LABELS))
            self.leakage_head = torch.nn.Linear(h, 4)
            self.action_head = torch.nn.Linear(h, len(ACTION_LABELS))
        def forward(self, pooled):
            return (
                self.attack_head(pooled),
                self.leakage_head(pooled),
                self.action_head(pooled),
            )

    head = MultiTaskHead(hidden)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base.to(device); head.to(device)

    # ---- Optimizer ----
    optim = torch.optim.AdamW(
        list(base.parameters()) + list(head.parameters()), lr=learning_rate,
    )
    total_steps = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )
    ce = torch.nn.CrossEntropyLoss()

    # ---- Train ----
    for ep in range(epochs):
        base.train(); head.train()
        for step, batch in enumerate(train_loader):
            optim.zero_grad()
            out = base(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            # mean-pool over non-padded tokens
            mask = batch["attention_mask"].to(device).unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)

            atk_l, lkg_l, act_l = head(pooled.float())
            loss = (
                ce(atk_l, batch["attack_label"].to(device))
                + ce(lkg_l, batch["leakage_label"].to(device))
                + ce(act_l, batch["action_label"].to(device))
            )
            loss.backward(); optim.step(); sched.step()
            if step % 50 == 0:
                logger.info("ep %d step %d loss %.4f", ep, step, loss.item())

        # ---- Eval ----
        base.eval(); head.eval()
        a_correct = l_correct = act_correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                out = base(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                mask = batch["attention_mask"].to(device).unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
                atk_l, lkg_l, act_l = head(pooled.float())

                a_correct  += (atk_l.argmax(-1).cpu() == batch["attack_label"]).sum().item()
                l_correct  += (lkg_l.argmax(-1).cpu() == batch["leakage_label"]).sum().item()
                act_correct+= (act_l.argmax(-1).cpu() == batch["action_label"]).sum().item()
                total += len(batch["attack_label"])

        logger.info(
            "ep %d val: attack=%.3f leakage=%.3f action=%.3f",
            ep, a_correct/total, l_correct/total, act_correct/total,
        )

    # ---- Save ----
    base.save_pretrained(output_dir / "lora")
    torch.save(head.state_dict(), output_dir / "head.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    # Final metrics
    final = {
        "n_train": len(train_ex), "n_val": len(val_ex),
        "attack_accuracy": a_correct/total if total else 0,
        "leakage_accuracy": l_correct/total if total else 0,
        "action_accuracy": act_correct/total if total else 0,
        "model_name": model_name,
    }
    (output_dir / "metrics.json").write_text(json.dumps(final, indent=2))
    logger.info("Validator saved to %s", output_dir)
    return final


# ---------------------------------------------------------------------------
# Step 3: Inference wrapper
# ---------------------------------------------------------------------------


class Validator:
    """Loaded validator usable by D4 defense.

    The defense calls validator.predict(history, response) and looks at
    `leakage_severity`. If >=2, regenerate with hardened prompt.
    """

    def __init__(self, model, head, tokenizer, device: str):
        self.model = model
        self.head = head
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(cls, checkpoint_dir: str | Path,
             base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(f"Validator requires torch/transformers/peft. {e}")
        checkpoint_dir = Path(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")
        base = AutoModel.from_pretrained(
            base_model, trust_remote_code=True,
            torch_dtype="bfloat16" if __import__("torch").cuda.is_available() else "float32",
        )
        model = PeftModel.from_pretrained(base, checkpoint_dir / "lora")
        head_state = __import__("torch").load(
            checkpoint_dir / "head.pt", map_location="cpu",
        )
        # Reconstruct head
        import torch
        hidden = base.config.hidden_size
        head = torch.nn.Module()
        head.attack_head = torch.nn.Linear(hidden, len(ATTACK_LABELS))
        head.leakage_head = torch.nn.Linear(hidden, 4)
        head.action_head = torch.nn.Linear(hidden, len(ACTION_LABELS))
        head.load_state_dict(head_state)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(model.to(device).eval(), head.to(device).eval(),
                   tokenizer, device)

    def predict(self, history: list[dict], response: str) -> dict:
        import torch
        # Build the same prompt format as training
        hist_text = "\n\n".join(
            f"Student: {m['content']}" if m["role"] == "user"
            else f"Tutor: {m['content']}"
            for m in history[-4:]
        )
        prompt = (
            "Classify the tutor response below.\n\n"
            f"Recent dialogue:\n{hist_text[:1500]}\n\n"
            f"Tutor response:\n{response[:1500]}"
        )
        enc = self.tokenizer(
            prompt, truncation=True, max_length=1024,
            padding="max_length", return_tensors="pt",
        )
        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
            mask = enc["attention_mask"].to(self.device).unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            atk_l = self.head.attack_head(pooled.float())
            lkg_l = self.head.leakage_head(pooled.float())
            act_l = self.head.action_head(pooled.float())
        return {
            "attack_type": ATTACK_LABELS[atk_l.argmax(-1).item()],
            "leakage_severity": int(lkg_l.argmax(-1).item()),
            "suggested_action": ACTION_LABELS[act_l.argmax(-1).item()],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="prepare training data")
    pp.add_argument("--scores", required=True)
    pp.add_argument("--dialogues", required=True)
    pp.add_argument("--output", required=True)
    pp.add_argument("--skip-d3-d4", action="store_true")

    pt = sub.add_parser("train", help="train validator")
    pt.add_argument("--train", required=True)
    pt.add_argument("--output", required=True)
    pt.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    pt.add_argument("--epochs", type=int, default=3)
    pt.add_argument("--batch-size", type=int, default=8)
    pt.add_argument("--lr", type=float, default=5e-5)

    args = p.parse_args()

    if args.cmd == "prepare":
        prepare_training_data(args.scores, args.dialogues, args.output,
                              skip_d3_d4=args.skip_d3_d4)
    elif args.cmd == "train":
        metrics = train_validator(
            args.train, args.output,
            model_name=args.model, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.lr,
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _main()
