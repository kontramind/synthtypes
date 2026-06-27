"""Verification plots and KS tests for the oracle generative process.

Run via scripts/verify_oracle.py on a 50k-sample draw.
All plots saved to outputs/verification/ as PNG.
All KS results printed to stdout with pass/fail.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


_OUT = Path("outputs/verification")
_N_VERIFY = 50_000
_KS_ALPHA = 0.05


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _ks_report(label: str, stat: float, pval: float) -> None:
    result = "PASS" if pval >= _KS_ALPHA else "FAIL"
    print(f"  [{result}] KS {label}: stat={stat:.4f}  p={pval:.4f}")


# ---------------------------------------------------------------------------
# 1. Marginal checks
# ---------------------------------------------------------------------------

def check_marginals(df) -> None:
    print("\n=== 1. Marginal checks ===")

    stat, pval = stats.kstest(df["x1"], stats.beta(2, 5).cdf)
    _ks_report("x1 ~ Beta(2,5)", stat, pval)

    stat, pval = stats.kstest(df["x2"], stats.gamma(3, scale=2).cdf)
    _ks_report("x2 ~ Gamma(3,2)", stat, pval)

    x3_mean = df["x3"].mean()
    result = "PASS" if abs(x3_mean - 0.3) < 0.01 else "FAIL"
    print(f"  [{result}] x3 mean={x3_mean:.4f}  (expected 0.3)")

    props = df["x6"].value_counts(normalize=True).sort_index()
    expected = {0: 0.5, 1: 0.3, 2: 0.2}
    for k, exp in expected.items():
        got = props.get(k, 0.0)
        result = "PASS" if abs(got - exp) < 0.01 else "FAIL"
        print(f"  [{result}] x6={k} prop={got:.4f}  (expected {exp})")


# ---------------------------------------------------------------------------
# 2. Residual checks
# ---------------------------------------------------------------------------

def check_residuals(df) -> None:
    print("\n=== 2. Residual checks ===")

    # x4 residual
    r4 = df["x4"] - (df["x1"] + df["x2"])
    stat, pval = stats.kstest(r4, stats.norm(0, 0.05).cdf)
    _ks_report("x4 residual ~ N(0, 0.05)", stat, pval)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r4, bins=80, density=True, alpha=0.7, label="x4 residual")
    xs = np.linspace(r4.min(), r4.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, 0, 0.05), "r-", lw=2, label="N(0, 0.05)")
    ax.set_title("x4 residual: x4 − (x1+x2)")
    ax.legend()
    _save(fig, "residual_x4")

    # x5 residual
    expected_x5 = df["x3"] * df["x1"] + (1 - df["x3"]) * df["x2"]
    r5 = df["x5"] - expected_x5
    stat, pval = stats.kstest(r5, stats.norm(0, 0.08).cdf)
    _ks_report("x5 residual ~ N(0, 0.08)", stat, pval)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r5, bins=80, density=True, alpha=0.7, label="x5 residual")
    xs = np.linspace(r5.min(), r5.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, 0, 0.08), "r-", lw=2, label="N(0, 0.08)")
    ax.set_title("x5 residual: x5 − E[x5|x3]")
    ax.legend()
    _save(fig, "residual_x5")

    # x7 residual per severity
    sigma_map = {0: 0.05, 1: 0.08, 2: 0.12}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for k, ax in zip([0, 1, 2], axes):
        mask = df["x6"] == k
        r7k = df.loc[mask, "x7"] - (k + 1) * df.loc[mask, "x1"]
        stat, pval = stats.kstest(r7k, stats.norm(0, sigma_map[k]).cdf)
        _ks_report(f"x7 residual (x6={k}) ~ N(0, {sigma_map[k]})", stat, pval)
        ax.hist(r7k, bins=80, density=True, alpha=0.7, label=f"x6={k}")
        xs = np.linspace(r7k.min(), r7k.max(), 300)
        ax.plot(xs, stats.norm.pdf(xs, 0, sigma_map[k]), "r-", lw=2,
                label=f"N(0, {sigma_map[k]})")
        ax.set_title(f"x7 residual | x6={k}")
        ax.legend()
    fig.tight_layout()
    _save(fig, "residual_x7_per_severity")


# ---------------------------------------------------------------------------
# 3. Conditional structure checks
# ---------------------------------------------------------------------------

def check_conditional_structure(df) -> None:
    print("\n=== 3. Conditional structure checks ===")

    sample = df.sample(n=min(5000, len(df)), random_state=0)

    # E[x5 | x3=0] tracks x2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    s0 = sample[sample["x3"] == 0]
    axes[0].scatter(s0["x2"], s0["x5"], alpha=0.2, s=5)
    lim = max(s0["x2"].max(), s0["x5"].max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", lw=1.5, label="y=x")
    axes[0].set_xlabel("x2")
    axes[0].set_ylabel("x5")
    axes[0].set_title("E[x5 | x3=0] ≈ x2")
    axes[0].legend()

    # E[x5 | x3=1] tracks x1
    s1 = sample[sample["x3"] == 1]
    axes[1].scatter(s1["x1"], s1["x5"], alpha=0.3, s=5)
    lim = max(s1["x1"].max(), s1["x5"].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "r--", lw=1.5, label="y=x")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x5")
    axes[1].set_title("E[x5 | x3=1] ≈ x1")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, "conditional_x5")

    # E[x7 | x6=k] = (k+1)*x1
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for k, ax in zip([0, 1, 2], axes):
        sk = sample[sample["x6"] == k]
        ax.scatter(sk["x1"], sk["x7"], alpha=0.2, s=5)
        xs = np.linspace(0, sk["x1"].max() * 1.05, 100)
        ax.plot(xs, (k + 1) * xs, "r--", lw=1.5, label=f"y={(k+1)}x1")
        ax.set_xlabel("x1")
        ax.set_ylabel("x7")
        ax.set_title(f"E[x7 | x6={k}] = {k+1}·x1")
        ax.legend()
    fig.tight_layout()
    _save(fig, "conditional_x7_per_severity")

    print("  Scatter plots saved.")


# ---------------------------------------------------------------------------
# 4. Noise floor plot
# ---------------------------------------------------------------------------

def noise_floor_plot(df) -> None:
    print("\n=== 4. Noise floor plot ===")

    r4 = (df["x4"] - (df["x1"] + df["x2"])).abs()

    expected_x5 = df["x3"] * df["x1"] + (1 - df["x3"]) * df["x2"]
    r5 = (df["x5"] - expected_x5).abs()

    # x7 has three sigma levels; show combined residuals with a threshold line per level
    r7 = (df["x7"] - (df["x6"] + 1) * df["x1"]).abs()

    # (sigma, residuals, title, [(threshold_label, value, color), ...])
    _sigma_x7 = [0.05, 0.08, 0.12]
    specs = [
        (0.05, r4, "|x4 − (x1+x2)|",
         [("3σ", 3 * 0.05, "orange"), ("6σ", 6 * 0.05, "red")]),
        (0.08, r5, "|x5 − E[x5|x3]|",
         [("3σ", 3 * 0.08, "orange"), ("6σ", 6 * 0.08, "red")]),
        (max(_sigma_x7), r7, "|x7 − (x6+1)·x1|  (all severities)",
         [(f"3σ(k={k})", 3 * s, c) for k, s, c in zip(
             [0, 1, 2], _sigma_x7, ["#f4a340", "#e07b00", "#b85c00"]
         )] + [
             (f"6σ(k={k})", 6 * s, c) for k, s, c in zip(
                 [0, 1, 2], _sigma_x7, ["#d44", "#a00", "#600"]
             )
         ]),
    ]

    # constrained_layout handles suptitle without needing tight_layout()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for (sigma, residuals, title, thresholds), ax in zip(specs, axes):
        xlim_max = 8 * sigma
        ax.hist(
            residuals.clip(upper=xlim_max),
            bins=100,
            range=(0, xlim_max),
            density=True,
            alpha=0.7,
        )
        for tlabel, tval, tcolor in thresholds:
            ax.axvline(tval, color=tcolor, lw=1.5, linestyle="--",
                       label=f"{tlabel} = {tval:.4f}")
        ax.set_xlim(0, xlim_max)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("absolute residual")
        ax.legend(fontsize=7)

    fig.suptitle("Noise floor — candidate epsilon thresholds", fontsize=13)
    _save(fig, "noise_floor")

    out_path = Path("outputs/verification/noise_floor.png")
    assert out_path.exists(), "noise_floor.png was not saved"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all(df) -> None:
    check_marginals(df)
    check_residuals(df)
    check_conditional_structure(df)
    noise_floor_plot(df)
    print(f"\nAll plots saved to {_OUT}/")
