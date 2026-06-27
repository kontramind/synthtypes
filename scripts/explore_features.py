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
