import numpy as np
import pandas as pd


def generate_oracle(n: int, seed: int | None = None) -> pd.DataFrame:
    """Generate n samples from the oracle generative process.

    Returns DataFrame with columns [x1, x2, x3, x4, x5, x6, x7].
    """
    rng = np.random.default_rng(seed)

    # Tier 1 — independent roots
    x1 = rng.beta(2, 5, size=n)                                      # Beta(2, 5)
    x2 = rng.gamma(shape=3, scale=2, size=n)                         # Gamma(3, 2)
    x3 = rng.binomial(1, 0.3, size=n).astype(float)                  # Bernoulli(0.3)
    x6 = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2]).astype(float)  # Categorical

    # Tier 2 — conditional on Tier 1
    x4 = x1 + x2 + rng.normal(0, 0.05, size=n)

    x5 = x3 * x1 + (1 - x3) * x2 + rng.normal(0, 0.08, size=n)

    sigma7 = np.where(x6 == 0, 0.05, np.where(x6 == 1, 0.08, 0.12))
    x7 = (x6 + 1) * x1 + rng.normal(0, 1, size=n) * sigma7

    return pd.DataFrame(
        {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6, "x7": x7}
    )
