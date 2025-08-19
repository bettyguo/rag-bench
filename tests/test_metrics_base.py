"""Tests for the bootstrap CI machinery in rag_bench.metrics.base."""

from __future__ import annotations

import numpy as np
import pytest

from rag_bench.metrics.base import bootstrap_ci, paired_bootstrap_p_value


def test_bootstrap_ci_contains_true_mean_for_normal_samples():
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=0.7, scale=0.1, size=500)
    lo, hi = bootstrap_ci(samples, n_bootstrap=2000, seed=0)
    assert lo < 0.7 < hi
    # CI half-width should be modest at n=500
    assert (hi - lo) < 0.05


def test_bootstrap_ci_degenerate_when_all_same():
    samples = np.full(50, 0.42)
    lo, hi = bootstrap_ci(samples, n_bootstrap=500, seed=0)
    assert lo == pytest.approx(0.42)
    assert hi == pytest.approx(0.42)


def test_bootstrap_ci_single_sample():
    samples = np.array([0.5])
    lo, hi = bootstrap_ci(samples)
    assert lo == hi == 0.5


def test_bootstrap_ci_empty_returns_nan():
    samples = np.array([], dtype=np.float64)
    lo, hi = bootstrap_ci(samples)
    assert np.isnan(lo) and np.isnan(hi)


def test_bca_method_runs():
    rng = np.random.default_rng(7)
    samples = rng.beta(2, 5, size=200)  # skewed
    lo_p, hi_p = bootstrap_ci(samples, method="percentile", n_bootstrap=2000, seed=0)
    lo_b, hi_b = bootstrap_ci(samples, method="bca", n_bootstrap=2000, seed=0)
    # both should bracket the sample mean
    m = samples.mean()
    assert lo_p < m < hi_p
    assert lo_b < m < hi_b


def test_paired_bootstrap_p_value_high_when_no_diff():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, size=100)
    b = a.copy()
    p = paired_bootstrap_p_value(a, b, n_bootstrap=2000, seed=0)
    assert p > 0.5


def test_paired_bootstrap_p_value_low_when_clear_diff():
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.05, size=200)
    b = rng.normal(0.5, 0.05, size=200)
    p = paired_bootstrap_p_value(a, b, n_bootstrap=2000, seed=0)
    assert p < 0.05


def test_paired_bootstrap_requires_equal_shape():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        paired_bootstrap_p_value(a, b)
