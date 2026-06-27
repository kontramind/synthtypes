"""Step 2 verification: runs all oracle checks on a 50k sample."""

from pathlib import Path

from synthtypes.oracle.generative import generate_oracle
from synthtypes.verification.plots import run_all, _N_VERIFY

Path("outputs/verification").mkdir(parents=True, exist_ok=True)

print(f"Generating {_N_VERIFY:,} samples from oracle (seed=3)...")
df = generate_oracle(_N_VERIFY, seed=3)
print(f"Shape: {df.shape}\n")

run_all(df)
