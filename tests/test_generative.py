import numpy as np
import pandas as pd
import pytest

from synthtypes.oracle.generative import generate_oracle


def test_output_shape():
    df = generate_oracle(100)
    assert df.shape == (100, 7)


def test_column_names():
    df = generate_oracle(10)
    assert list(df.columns) == ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]


def test_x3_binary():
    df = generate_oracle(1000, seed=0)
    assert set(df["x3"].unique()).issubset({0.0, 1.0})


def test_x6_values():
    df = generate_oracle(1000, seed=0)
    assert set(df["x6"].unique()).issubset({0.0, 1.0, 2.0})


def test_x1_support():
    df = generate_oracle(1000, seed=0)
    assert (df["x1"] > 0).all() and (df["x1"] < 1).all()


def test_x2_support():
    df = generate_oracle(1000, seed=0)
    assert (df["x2"] > 0).all()


def test_reproducibility():
    df_a = generate_oracle(500, seed=7)
    df_b = generate_oracle(500, seed=7)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_x4_residual_stats():
    df = generate_oracle(50_000, seed=1)
    residual = df["x4"] - (df["x1"] + df["x2"])
    assert abs(residual.mean()) < 0.05 * 1.2
    assert abs(residual.std() - 0.05) < 0.05 * 0.2


def test_x5_residual_stats():
    df = generate_oracle(50_000, seed=2)
    expected = df["x3"] * df["x1"] + (1 - df["x3"]) * df["x2"]
    residual = df["x5"] - expected
    assert abs(residual.mean()) < 0.08 * 1.2
    assert abs(residual.std() - 0.08) < 0.08 * 0.2
