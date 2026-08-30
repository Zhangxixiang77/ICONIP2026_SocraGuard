"""Statistical analysis for the paper.

Provides:
- Bootstrap 95% confidence intervals for ASR
- Fisher's exact test for D2-vs-D3 (and any defense pairs)
- Cohen's kappa (multi-rater for human evaluation)
- McNemar's test (for paired same-scenario comparisons)
- ASR (95% CI) reporting helpers

Usage:
  from src.stats import bootstrap_ci, fisher_pair, cohens_kappa
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ci(
    leakage_full: np.ndarray,
    leakage_partial: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for ASR = mean(leaked_full) + 0.5 * mean(leaked_partial).

    Inputs are 1-d boolean / 0-1 arrays of length N (per-dialogue indicators).
    """
    rng = np.random.RandomState(seed)
    leakage_full = np.asarray(leakage_full).astype(float)
    leakage_partial = np.asarray(leakage_partial).astype(float)
    n = len(leakage_full)
    assert n == len(leakage_partial)

    point_asr = float(leakage_full.mean() + 0.5 * leakage_partial.mean())
    samples = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        samples[b] = leakage_full[idx].mean() + 0.5 * leakage_partial[idx].mean()

    alpha = 1.0 - confidence
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {
        "asr": point_asr,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": n,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Fisher exact test for paired defense comparisons
# ---------------------------------------------------------------------------


def fisher_pair(
    leaked_a: np.ndarray,
    leaked_b: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> dict:
    """Fisher exact test on independent samples of binary 'leaked' outcomes.

    leaked_a: defense A's per-scenario binary leakage (0/1)
    leaked_b: defense B's per-scenario binary leakage (0/1)
    """
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        raise ImportError("Install scipy: pip install scipy")

    a_leaked = int(np.sum(leaked_a)); a_total = len(leaked_a)
    b_leaked = int(np.sum(leaked_b)); b_total = len(leaked_b)
    table = [[a_leaked, a_total - a_leaked],
             [b_leaked, b_total - b_leaked]]
    odds, p = fisher_exact(table, alternative=alternative)
    return {
        "a_leak_rate": a_leaked / a_total if a_total else 0.0,
        "b_leak_rate": b_leaked / b_total if b_total else 0.0,
        "odds_ratio": float(odds),
        "p_value": float(p),
        "n_a": a_total, "n_b": b_total,
    }


# ---------------------------------------------------------------------------
# McNemar (paired)
# ---------------------------------------------------------------------------


def mcnemar(
    leaked_a: np.ndarray,
    leaked_b: np.ndarray,
) -> dict:
    """McNemar's test for paired binary outcomes.

    Use when the same scenarios are scored under two defenses
    (which is our case for D2 vs D3 on the same set of scenarios).
    """
    try:
        from scipy.stats import binomtest
    except ImportError:
        try:
            from scipy.stats import binom_test as binomtest  # legacy
        except ImportError:
            raise ImportError("Install scipy: pip install scipy")

    leaked_a = np.asarray(leaked_a).astype(int)
    leaked_b = np.asarray(leaked_b).astype(int)
    assert len(leaked_a) == len(leaked_b)

    b01 = int(np.sum((leaked_a == 0) & (leaked_b == 1)))   # only B leaked
    b10 = int(np.sum((leaked_a == 1) & (leaked_b == 0)))   # only A leaked
    n_disc = b01 + b10
    if n_disc == 0:
        return {"b01": 0, "b10": 0, "n_discordant": 0, "p_value": 1.0}
    try:
        result = binomtest(b10, n=n_disc, p=0.5, alternative="two-sided")
        p = float(result.pvalue)
    except (TypeError, AttributeError):
        # fallback for older scipy where binomtest != binom_test
        from scipy.stats import binom
        p = 2 * min(binom.cdf(min(b01, b10), n_disc, 0.5),
                    1 - binom.cdf(min(b01, b10) - 1, n_disc, 0.5))
        p = float(min(p, 1.0))
    return {"b01": b01, "b10": b10, "n_discordant": n_disc, "p_value": p}


# ---------------------------------------------------------------------------
# Cohen's kappa (per-pair, multi-class ordinal)
# ---------------------------------------------------------------------------


def cohens_kappa(
    rater_a: np.ndarray,
    rater_b: np.ndarray,
    *,
    weighted: str | None = None,  # None | "linear" | "quadratic"
    levels: list | None = None,
) -> dict:
    """Cohen's (weighted) kappa.

    weighted=None    : standard kappa (treats every disagreement equally).
    weighted="linear": linear weights (good for ordinal scales).
    """
    rater_a = np.asarray(rater_a)
    rater_b = np.asarray(rater_b)
    assert len(rater_a) == len(rater_b)

    if levels is None:
        levels = sorted(set(np.concatenate([rater_a, rater_b]).tolist()))
    K = len(levels)
    idx = {v: i for i, v in enumerate(levels)}
    O = np.zeros((K, K), dtype=int)
    for a, b in zip(rater_a, rater_b):
        O[idx[a], idx[b]] += 1

    total = O.sum()
    pa = O.sum(axis=1) / total
    pb = O.sum(axis=0) / total
    E = np.outer(pa, pb) * total

    if weighted is None:
        po = O.diagonal().sum() / total
        pe = E.diagonal().sum() / total
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        return {"kappa": float(kappa), "weighted": "none", "K": K, "n": int(total)}

    W = np.zeros((K, K))
    if weighted == "linear":
        for i in range(K):
            for j in range(K):
                W[i, j] = abs(i - j) / (K - 1)
    elif weighted == "quadratic":
        for i in range(K):
            for j in range(K):
                W[i, j] = ((i - j) ** 2) / ((K - 1) ** 2)
    else:
        raise ValueError(f"weighted must be None|linear|quadratic, got {weighted}")

    kappa = 1 - (W * O).sum() / (W * E).sum() if (W * E).sum() > 0 else 0.0
    return {"kappa": float(kappa), "weighted": weighted, "K": K, "n": int(total)}


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------


@dataclass
class CellStats:
    tutor: str
    defense: str
    n: int
    asr: float
    asr_ci: tuple[float, float]


def cells_with_ci(df, n_bootstrap: int = 5000, seed: int = 42) -> list[CellStats]:
    """Compute ASR + bootstrap CI per (tutor, defense) cell from per-dialogue df.

    Expects df with columns: tutor_backend, defense, leaked_full, leaked_partial.
    """
    out = []
    for (tutor, defense), g in df.groupby(["tutor_backend", "defense"]):
        ci = bootstrap_ci(
            g["leaked_full"].values,
            g["leaked_partial"].values,
            n_bootstrap=n_bootstrap, seed=seed,
        )
        out.append(CellStats(
            tutor=tutor, defense=defense, n=len(g),
            asr=ci["asr"], asr_ci=(ci["ci_low"], ci["ci_high"]),
        ))
    return out


def pretty_table_with_ci(cells: list[CellStats]) -> str:
    """Format ASR (95% CI) cells for paper. Returns a string table."""
    out_lines = []
    out_lines.append(f"{'Tutor':<20} {'Defense':<10} {'N':>4} {'ASR (95% CI)':>22}")
    out_lines.append("-" * 60)
    for c in sorted(cells, key=lambda c: (c.tutor, c.defense)):
        ci_str = f"{c.asr*100:5.1f} [{c.asr_ci[0]*100:4.1f}, {c.asr_ci[1]*100:4.1f}]"
        out_lines.append(f"{c.tutor:<20} {c.defense:<10} {c.n:>4} {ci_str:>22}")
    return "\n".join(out_lines)
