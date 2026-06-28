"""
Feature distribution explorer.
Plots oracle feature distributions for inspection and documentation.

Note: the draw here (50k, seed=42) is for visualisation only.
It is NOT the experimental population P (100k, generated in Step 4).

Run:  uv run python scripts/explore_features.py
Plots saved to outputs/exploration/
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from synthtypes.oracle.generative import generate_oracle


OUT = Path("outputs/exploration")
OUT.mkdir(parents=True, exist_ok=True)

df = generate_oracle(50_000, seed=42)

# ---------------------------------------------------------------------------
# x1 distribution plot
# ---------------------------------------------------------------------------

x1 = df["x1"]
x1_mean = x1.mean()
x1_median = x1.median()
p5, p25, p75, p95 = x1.quantile([0.05, 0.25, 0.75, 0.95])

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(x1, bins=80, density=True, alpha=0.7, label="observed")

xs = np.linspace(0, 1, 500)
ax.plot(xs, stats.beta(2, 5).pdf(xs), color="red", lw=2, label="Beta(2, 5) PDF")

ax.axvline(x1_mean, color="navy", lw=1.5, linestyle="--",
           label=f"mean = {x1_mean:.3f}")
ax.axvline(x1_median, color="green", lw=1.5, linestyle=":",
           label=f"median = {x1_median:.3f}")

ax.axvspan(p5, p95, color="lightblue", alpha=0.15, label="5th–95th pct")

ax.set_xlabel("x1 (bounded physiological index)")
ax.set_ylabel("density")
ax.set_title("x1 ~ Beta(2, 5)")

ax.text(
    0.98, 0.97,
    "support: (0, 1)  |  mean ≈ 0.286  |  mode ≈ 0.167",
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
)

ax.legend(fontsize=8)
fig.tight_layout()

out_path = OUT / "x1_distribution.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

assert out_path.exists(), "x1_distribution.png was not saved"
print(f"Saved → {out_path}")

# ---------------------------------------------------------------------------
# x1 summary statistics
# ---------------------------------------------------------------------------

print("\nx1 summary statistics:")
print(f"  mean:   {x1.mean():.4f}")
print(f"  median: {x1.median():.4f}")
print(f"  std:    {x1.std():.4f}")
print(f"  5th pct:  {p5:.4f}")
print(f"  25th pct: {p25:.4f}")
print(f"  75th pct: {p75:.4f}")
print(f"  95th pct: {p95:.4f}")
print(f"  min:    {x1.min():.4f}")
print(f"  max:    {x1.max():.4f}")

# ---------------------------------------------------------------------------
# x2 distribution plot
# ---------------------------------------------------------------------------

x2 = df["x2"]
x2_mean = x2.mean()
x2_median = x2.median()
x2_p5, x2_p25, x2_p75, x2_p95 = x2.quantile([0.05, 0.25, 0.75, 0.95])

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(x2, bins=80, density=True, alpha=0.7, label="observed")

xs2 = np.linspace(0, x2.max() * 1.05, 500)
ax.plot(xs2, stats.gamma(3, scale=2).pdf(xs2), color="red", lw=2,
        label="Gamma(3, 2) PDF")

ax.axvline(x2_mean, color="navy", lw=1.5, linestyle="--",
           label=f"mean = {x2_mean:.3f}")
ax.axvline(x2_median, color="green", lw=1.5, linestyle=":",
           label=f"median = {x2_median:.3f}")

ax.axvspan(x2_p5, x2_p95, color="lightblue", alpha=0.15, label="5th–95th pct")

ax.set_xlabel("x2 (unbounded positive measure)")
ax.set_ylabel("density")
ax.set_title("x2 ~ Gamma(3, 2)")

ax.text(
    0.98, 0.97,
    "support: (0, ∞)  |  mean ≈ 6.0  |  mode ≈ 4.0",
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
)

ax.legend(fontsize=8)
fig.tight_layout()

out_path_x2 = OUT / "x2_distribution.png"
fig.savefig(out_path_x2, dpi=150, bbox_inches="tight")
plt.close(fig)

assert out_path_x2.exists(), "x2_distribution.png was not saved"
print(f"\nSaved → {out_path_x2}")

# ---------------------------------------------------------------------------
# x2 summary statistics
# ---------------------------------------------------------------------------

print("\nx2 summary statistics:")
print(f"  mean:   {x2_mean:.4f}")
print(f"  median: {x2_median:.4f}")
print(f"  std:    {x2.std():.4f}")
print(f"  5th pct:  {x2_p5:.4f}")
print(f"  25th pct: {x2_p25:.4f}")
print(f"  75th pct: {x2_p75:.4f}")
print(f"  95th pct: {x2_p95:.4f}")
print(f"  min:    {x2.min():.4f}")
print(f"  max:    {x2.max():.4f}")

# ---------------------------------------------------------------------------
# x3 distribution plot
# ---------------------------------------------------------------------------

x3 = df["x3"]
x3_counts = x3.value_counts().sort_index()
x3_props = x3.value_counts(normalize=True).sort_index()

fig, ax = plt.subplots(figsize=(6, 5))

bars = ax.bar([0, 1], x3_props.values, width=0.4, alpha=0.8,
              color=["steelblue", "salmon"], label="observed proportion")

for bar, prop in zip(bars, x3_props.values):
    ax.annotate(
        f"{prop * 100:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=10,
    )

ax.axhline(0.7, color="blue", lw=1.5, linestyle="--", label="p(x3=0) = 0.7")
ax.axhline(0.3, color="red", lw=1.5, linestyle="--", label="p(x3=1) = 0.3")

ax.set_xticks([0, 1])
ax.set_xticklabels(["0 (majority)", "1 (minority)"])
ax.set_ylabel("proportion")
ax.set_title("x3 ~ Bernoulli(p=0.3)")
ax.set_ylim(0, 1.0)

ax.text(
    0.98, 0.97,
    "support: {0, 1}  |  p(x3=1) = 0.3  |  routing switch for x5",
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
)

ax.legend(fontsize=8)
fig.tight_layout()

out_path_x3 = OUT / "x3_distribution.png"
fig.savefig(out_path_x3, dpi=150, bbox_inches="tight")
plt.close(fig)

assert out_path_x3.exists(), "x3_distribution.png was not saved"
print(f"\nSaved → {out_path_x3}")

# ---------------------------------------------------------------------------
# x3 summary statistics
# ---------------------------------------------------------------------------

print("\nx3 summary statistics:")
for v in [0, 1]:
    count = int(x3_counts.get(float(v), 0))
    prop = x3_props.get(float(v), 0.0)
    print(f"  x3={v}: count={count}  proportion={prop:.4f}")
print(f"  deviation from p=0.3: {abs(x3_props.get(1.0, 0.0) - 0.3):.4f}")
