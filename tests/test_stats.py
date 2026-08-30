"""Unit tests for src/stats.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats import (
    bootstrap_ci, cohens_kappa, fisher_pair, mcnemar,
)


def test_bootstrap_ci_zero():
    full = np.zeros(100); partial = np.zeros(100)
    r = bootstrap_ci(full, partial, n_bootstrap=500, seed=0)
    assert r["asr"] == 0.0
    assert r["ci_low"] == 0.0
    assert r["ci_high"] == 0.0
    print("✓ bootstrap_ci returns 0 CI when no leakage")


def test_bootstrap_ci_all():
    full = np.ones(50); partial = np.zeros(50)
    r = bootstrap_ci(full, partial, n_bootstrap=500, seed=0)
    assert r["asr"] == 1.0
    assert r["ci_low"] == 1.0 and r["ci_high"] == 1.0
    print("✓ bootstrap_ci returns 1 CI when all full leakage")


def test_bootstrap_ci_mid():
    rng = np.random.RandomState(0)
    full = (rng.random(200) < 0.3).astype(int)
    partial = (rng.random(200) < 0.2).astype(int) * (1 - full)
    r = bootstrap_ci(full, partial, n_bootstrap=2000, seed=42)
    asr_naive = float(full.mean() + 0.5 * partial.mean())
    assert abs(r["asr"] - asr_naive) < 1e-6
    assert r["ci_low"] <= r["asr"] <= r["ci_high"]
    width = r["ci_high"] - r["ci_low"]
    assert 0.02 < width < 0.30, f"unexpected CI width: {width}"
    print(f"✓ bootstrap_ci mid-range: ASR={r['asr']:.3f}, "
          f"CI=[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


def test_fisher_pair_significant():
    a = [1] * 80 + [0] * 20
    b = [1] * 20 + [0] * 80
    r = fisher_pair(np.array(a), np.array(b))
    assert r["p_value"] < 0.001
    assert r["a_leak_rate"] == 0.8
    assert r["b_leak_rate"] == 0.2
    print(f"✓ fisher_pair detects significant difference (p={r['p_value']:.2e})")


def test_fisher_pair_no_difference():
    a = [1] * 50 + [0] * 50
    b = [1] * 50 + [0] * 50
    r = fisher_pair(np.array(a), np.array(b))
    assert r["p_value"] > 0.5
    print(f"✓ fisher_pair: no diff -> p={r['p_value']:.3f}")


def test_mcnemar_paired():
    """30 scenarios where only B leaked, 5 where only A leaked"""
    a = [0] * 30 + [1] * 5 + [1] * 50 + [0] * 15
    b = [1] * 30 + [0] * 5 + [1] * 50 + [0] * 15
    r = mcnemar(np.array(a), np.array(b))
    assert r["b01"] == 30
    assert r["b10"] == 5
    assert r["p_value"] < 0.001
    print(f"✓ mcnemar: 30 vs 5 discordant -> p={r['p_value']:.2e}")


def test_mcnemar_zero_discordant():
    a = [1, 0, 1, 0]
    b = [1, 0, 1, 0]
    r = mcnemar(np.array(a), np.array(b))
    assert r["n_discordant"] == 0
    assert r["p_value"] == 1.0
    print("✓ mcnemar: identical raters -> p=1.0")


def test_cohens_kappa_perfect():
    rater = [0, 1, 2, 3, 0, 1, 2, 3] * 10
    r = cohens_kappa(np.array(rater), np.array(rater))
    assert r["kappa"] == 1.0
    print("✓ cohens_kappa: perfect agreement -> kappa=1.0")


def test_cohens_kappa_chance():
    rng = np.random.RandomState(0)
    a = rng.randint(0, 4, size=400)
    b = rng.randint(0, 4, size=400)
    r = cohens_kappa(a, b)
    assert -0.15 < r["kappa"] < 0.15
    print(f"✓ cohens_kappa: random agreement -> kappa={r['kappa']:.3f}")


def test_cohens_kappa_weighted():
    a = np.array([0, 1, 2, 3] * 10)
    b = np.array([1, 2, 3, 0] * 10)   # always off-by-one (cyclic)
    r_unw = cohens_kappa(a, b)
    r_lin = cohens_kappa(a, b, weighted="linear")
    # Cyclic disagreement makes weighted kappa lower than unweighted (rare but possible);
    # the main thing we check is that they differ when disagreement is nonzero.
    assert abs(r_unw["kappa"] - r_lin["kappa"]) > 1e-6
    print(f"✓ cohens_kappa weighted differs from unweighted: "
          f"unw={r_unw['kappa']:.3f}, lin={r_lin['kappa']:.3f}")


if __name__ == "__main__":
    print("Running stats unit tests...\n")
    test_bootstrap_ci_zero()
    test_bootstrap_ci_all()
    test_bootstrap_ci_mid()
    test_fisher_pair_significant()
    test_fisher_pair_no_difference()
    test_mcnemar_paired()
    test_mcnemar_zero_discordant()
    test_cohens_kappa_perfect()
    test_cohens_kappa_chance()
    test_cohens_kappa_weighted()
    print("\nAll stats tests passed ✓")
