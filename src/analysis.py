"""Tables and figures for the paper.

Produces (in --out directory):
- table_main.csv          : Main results (Table 2)
- table_main_with_ci.csv  : Same with 95% bootstrap CI
- table_attack_defense.csv: ASR per (defense, attack)
- table_cross_backend.csv : ASR per (defense, tutor)
- table_stagewise.csv     : Stage-wise leakage on multi-turn erosion
- table_kaplan_meier.csv  : KM survival values
- table_significance.csv  : Pairwise significance tests
- figure_heatmap.pdf      : Attack-defense heatmap
- figure_km_curve.pdf     : Kaplan-Meier survival curves
- figure_cross_backend.pdf: Same skill across LLMs
- summary.json            : Headline numbers, machine-readable
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import (
    compute_attack_x_defense,
    compute_kaplan_meier,
    compute_main_table,
    cross_backend_table,
    load_scores_dataframe,
)
from .stats import (
    bootstrap_ci, cells_with_ci, fisher_pair, mcnemar, pretty_table_with_ci,
)


# Publication-quality matplotlib defaults
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Defense visual style for KM curves
DEFENSE_COLOR = {
    "D0": "#E69F00", "D1": "#56B4E9", "D2": "#009E73",
    "D3": "#D55E00", "D4": "#0072B2",
}
DEFENSE_LINESTYLE = {
    "D0": (0, (1, 1)), "D1": (0, (3, 1, 1, 1)), "D2": (0, (5, 2)),
    "D3": "solid", "D4": "solid",
}
DEFENSE_LINEWIDTH = {
    "D0": 1.4, "D1": 1.4, "D2": 1.6, "D3": 2.4, "D4": 2.0,
}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _save_main_table(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    main = compute_main_table(df)
    rounded = main.copy()
    for col in ("ALR", "PLR", "ASR", "RQ", "PC"):
        rounded[col] = rounded[col].round(3)
    rounded.to_csv(out_dir / "table_main.csv", index=False)
    return rounded


def _save_main_with_ci(df: pd.DataFrame, out_dir: Path) -> str:
    cells = cells_with_ci(df)
    pretty = pretty_table_with_ci(cells)
    (out_dir / "table_main_with_ci.txt").write_text(pretty + "\n")
    rows = [{
        "tutor_backend": c.tutor, "defense": c.defense, "n": c.n,
        "asr": c.asr, "ci_low": c.asr_ci[0], "ci_high": c.asr_ci[1],
    } for c in cells]
    pd.DataFrame(rows).to_csv(out_dir / "table_main_with_ci.csv", index=False)
    return pretty


def _save_significance(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Pairwise D2-vs-D3 (and other key pairs) Fisher + McNemar tests."""
    rows = []
    for tutor in df["tutor_backend"].unique():
        sub = df[df["tutor_backend"] == tutor]
        # Pair D2 vs D3 on the same scenarios (paired analysis is best)
        d2 = sub[sub["defense"] == "D2"].set_index("scenario_id")
        d3 = sub[sub["defense"] == "D3"].set_index("scenario_id")
        common = d2.index.intersection(d3.index)
        if len(common) < 5:
            continue
        a = d2.loc[common, "leaked_full"].values + 0  # paired
        b = d3.loc[common, "leaked_full"].values + 0

        f = fisher_pair(a, b)
        m = mcnemar(a, b)
        rows.append({
            "tutor_backend": tutor,
            "comparison": "D2 vs D3 (full leakage only)",
            "n_paired": len(common),
            "d2_leak_rate": f["a_leak_rate"],
            "d3_leak_rate": f["b_leak_rate"],
            "fisher_p": f["p_value"],
            "mcnemar_p": m["p_value"],
            "mcnemar_b01_d3_only": m["b01"],
            "mcnemar_b10_d2_only": m["b10"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "table_significance.csv", index=False)
    return out


def _save_stagewise(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Stage-wise leakage rates from multi-turn erosion scenarios.

    For each defense and stage (A: turns 1-4, B: 5-6, C: 7-8, D: 9-10),
    compute the fraction of (defense × scenario) pairs where leakage_level >= 2
    occurs by the end of that stage.
    """
    # We need per-turn leakage info; load from scores dataframe NOTE this requires
    # the df to retain turn_idx — see metrics.py modifications.
    # For now this is computed externally if needed.
    sub = df[df["attack_class"] == "MULTI_TURN_EROSION"].copy()
    if sub.empty:
        return pd.DataFrame()

    stages = {
        "A": (1, 4),   # turns 1-4
        "B": (5, 6),
        "C": (7, 8),
        "D": (9, 10),
    }

    rows = []
    for defense, g in sub.groupby("defense"):
        row = {"defense": defense}
        for stage, (lo, hi) in stages.items():
            # crashed_by_end_of_stage = crash_turn <= hi AND crashed=1
            crashed_by_stage = ((g["crash_turn"] <= hi) & (g["crashed"] == 1)).mean()
            row[f"Stage_{stage}_leak"] = crashed_by_stage
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "table_stagewise.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save_heatmap(df: pd.DataFrame, out_dir: Path):
    ax_data = compute_attack_x_defense(df)
    ax_data.to_csv(out_dir / "table_attack_defense.csv")
    if ax_data.empty:
        return

    # Reorder
    defense_order = ["D0", "D1", "D2", "D3", "D4"]
    attack_order = [
        "DIRECT_DEMAND", "AUTHORITY_APPEAL", "ROLE_OVERRIDE",
        "HYPOTHETICAL_FRAMING", "EMOTIONAL_MANIPULATION",
        "PARTIAL_EXTRACTION", "REVERSE_PSYCHOLOGY", "MULTI_TURN_EROSION",
    ]
    ax_data = ax_data.reindex(
        index=[d for d in defense_order if d in ax_data.index],
        columns=[a for a in attack_order if a in ax_data.columns],
    )

    fig, ax_plot = plt.subplots(figsize=(8.5, 3.6))
    sns.heatmap(
        ax_data, annot=True, fmt=".2f", cmap="YlOrRd",
        cbar_kws={"label": "Attack Success Rate (ASR)"},
        ax=ax_plot, vmin=0, vmax=1, linewidths=0.4, linecolor="white",
    )
    ax_plot.set_xlabel("Attack Class")
    ax_plot.set_ylabel("Defense")
    plt.setp(ax_plot.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_heatmap.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_dir / "figure_heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _save_km_curve(df: pd.DataFrame, out_dir: Path):
    km = compute_kaplan_meier(df, attack_class="MULTI_TURN_EROSION")
    if not km:
        return

    fig, ax_plot = plt.subplots(figsize=(5.5, 3.6))

    # Stage shading
    STAGES = [(2, 4, "Stage A"), (4, 6, "Stage B"),
              (6, 8, "Stage C"), (8, 10, "Stage D")]
    for x0, x1, label in STAGES:
        ax_plot.axvspan(x0, x1, alpha=0.04, color="gray", zorder=0)
        ax_plot.text((x0 + x1) / 2, 1.04, label, ha="center", va="bottom",
                     fontsize=7.5, color="#666", style="italic", zorder=1)

    # Curves
    defense_order = ["D0", "D1", "D2", "D3", "D4"]
    for defense in defense_order:
        if defense not in km:
            continue
        dfk = km[defense]
        label = defense + (r"$^{*}$" if defense == "D3" else "")
        ax_plot.step(
            dfk["turn"], dfk["survival"],
            where="post", label=label,
            color=DEFENSE_COLOR.get(defense, "gray"),
            linestyle=DEFENSE_LINESTYLE.get(defense, "solid"),
            linewidth=DEFENSE_LINEWIDTH.get(defense, 1.5),
            zorder=3 if defense in ("D3", "D4") else 2,
        )

    # D2-vs-D3 gap arrow at turn 7
    if "D2" in km and "D3" in km:
        d2_7 = km["D2"][km["D2"]["turn"] == 7]["survival"].values
        d3_7 = km["D3"][km["D3"]["turn"] == 7]["survival"].values
        if len(d2_7) and len(d3_7):
            ax_plot.annotate("", xy=(7, d3_7[0]), xytext=(7, d2_7[0]),
                             arrowprops=dict(arrowstyle="<->", color="#444", lw=0.9,
                                             shrinkA=2, shrinkB=2), zorder=4)
            ax_plot.text(7.2, (d2_7[0] + d3_7[0]) / 2,
                         f"+{(d3_7[0]-d2_7[0])*100:.0f} pp",
                         fontsize=8.5, color="#222", va="center", style="italic")

    ax_plot.set_xlim(0, 10); ax_plot.set_ylim(0, 1.06)
    ax_plot.set_xticks(np.arange(0, 11))
    ax_plot.set_yticks(np.arange(0, 1.01, 0.2))
    ax_plot.set_xlabel("Turn")
    ax_plot.set_ylabel("Fraction of scenarios still robust\n(no leakage $\\geq$ partial)")
    ax_plot.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.4)
    ax_plot.set_axisbelow(True)
    legend = ax_plot.legend(title="Defense", loc="lower left",
                            bbox_to_anchor=(0.01, 0.02), frameon=True,
                            framealpha=0.92, edgecolor="#ccc")
    legend.get_frame().set_linewidth(0.6)
    ax_plot.text(0.99, 0.02, r"$^{*}$ours: per-turn re-injection",
                 transform=ax_plot.transAxes, ha="right", va="bottom",
                 fontsize=7.5, color="#444", style="italic")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "figure_km_curve.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_dir / "figure_km_curve.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # Save raw KM table
    km_combined = []
    for defense, dfk in km.items():
        d = dfk.copy(); d["defense"] = defense
        km_combined.append(d)
    if km_combined:
        pd.concat(km_combined).to_csv(out_dir / "table_kaplan_meier.csv", index=False)


def _save_cross_backend(df: pd.DataFrame, out_dir: Path):
    cb = cross_backend_table(df)
    cb.to_csv(out_dir / "table_cross_backend.csv")
    if cb.empty:
        return

    fig, ax_plot = plt.subplots(figsize=(7.5, 3.6))
    cb.plot(kind="bar", ax=ax_plot, edgecolor="white", linewidth=0.8)
    ax_plot.set_ylabel("Attack Success Rate (ASR)")
    ax_plot.set_xlabel("Defense")
    ax_plot.legend(title="Tutor backend", bbox_to_anchor=(1.02, 1),
                   loc="upper left", frameon=True, edgecolor="#ccc")
    ax_plot.tick_params(axis="x", rotation=0)
    ax_plot.grid(True, axis="y", alpha=0.3)
    ax_plot.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_cross_backend.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_dir / "figure_cross_backend.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_full_analysis(scores_path: str | Path, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_scores_dataframe(scores_path)
    df.to_csv(out_dir / "per_dialogue.csv", index=False)
    print(f"Loaded {len(df)} scored dialogues.")
    if df.empty:
        print("No data to analyze.")
        return

    print("\n=== Main results (Table 2 of the paper) ===")
    main = _save_main_table(df, out_dir)
    print(main.to_string(index=False))

    print("\n=== Main with bootstrap 95% CI ===")
    print(_save_main_with_ci(df, out_dir))

    print("\n=== Pairwise significance (D2 vs D3) ===")
    sig = _save_significance(df, out_dir)
    if not sig.empty:
        print(sig.to_string(index=False))

    print("\n=== Stage-wise erosion ===")
    stage = _save_stagewise(df, out_dir)
    if not stage.empty:
        print(stage.to_string(index=False))

    print("\n=== Generating figures ===")
    _save_heatmap(df, out_dir)
    _save_km_curve(df, out_dir)
    _save_cross_backend(df, out_dir)
    print(f"  -> figure_heatmap.pdf / .png")
    print(f"  -> figure_km_curve.pdf / .png")
    print(f"  -> figure_cross_backend.pdf / .png")

    # Headline numbers
    summary = {
        "n_dialogues": int(len(df)),
        "n_unique_scenarios": int(df["scenario_id"].nunique()),
        "tutors": sorted(df["tutor_backend"].unique().tolist()),
        "defenses": sorted(df["defense"].unique().tolist()),
        "attack_classes": sorted(df["attack_class"].unique().tolist()),
    }
    if {"D0", "D3"}.issubset(set(df["defense"].unique())):
        d0_asr = (df[df["defense"] == "D0"]["leaked_full"].mean()
                  + 0.5 * df[df["defense"] == "D0"]["leaked_partial"].mean())
        d3_asr = (df[df["defense"] == "D3"]["leaked_full"].mean()
                  + 0.5 * df[df["defense"] == "D3"]["leaked_partial"].mean())
        summary["headline_avg_asr"] = {
            "D0": float(d0_asr),
            "D3": float(d3_asr),
            "improvement_factor": float(d0_asr / d3_asr) if d3_asr > 0 else None,
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nAnalysis written to {out_dir.resolve()}")
