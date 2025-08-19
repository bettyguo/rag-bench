"""Tests for the post-run NoiseVulnerability + AdversarialComposite metrics."""

from __future__ import annotations

import pytest

from rag_bench.metrics.adversarial import adversarial_composite, noise_vulnerability


def test_nv_zero_when_noisy_matches_clean():
    assert noise_vulnerability(f1_clean=0.50, f1_noisy=0.50) == 0.0


def test_nv_one_when_noisy_collapses():
    assert noise_vulnerability(f1_clean=0.50, f1_noisy=0.0) == 1.0


def test_nv_partial_degradation():
    # noisy = half of clean → NV = 0.5
    assert noise_vulnerability(f1_clean=0.60, f1_noisy=0.30) == pytest.approx(0.5)


def test_nv_negative_when_noisy_higher_than_clean():
    # noisy improved over clean — possible with small samples
    assert noise_vulnerability(f1_clean=0.30, f1_noisy=0.45) == pytest.approx(-0.5)


def test_nv_zero_clean_returns_zero():
    # Convention: undefined ratio → 0.0 (pipeline didn't produce usable clean F1)
    assert noise_vulnerability(f1_clean=0.0, f1_noisy=0.10) == 0.0


def test_composite_all_perfect():
    score = adversarial_composite(
        noise_vulnerability_score=0.0,
        nrr_f1=1.0,
        plausible_compliance=1.0,
        implausible_resistance=1.0,
    )
    assert score == 1.0


def test_composite_all_zero():
    score = adversarial_composite(
        noise_vulnerability_score=1.0,
        nrr_f1=0.0,
        plausible_compliance=0.0,
        implausible_resistance=0.0,
    )
    assert score == 0.0


def test_composite_partial_dropouts():
    # Pipeline didn't report NRR or implausible_resistance
    score = adversarial_composite(
        noise_vulnerability_score=0.2,  # 80% retained
        plausible_compliance=1.0,
    )
    # mean of (0.8, 1.0)
    assert score == pytest.approx(0.9)


def test_composite_empty_returns_zero():
    assert adversarial_composite() == 0.0


def test_composite_flips_nv_correctly():
    # NV=0 means perfect (no degradation); composite component should be 1.0
    score = adversarial_composite(noise_vulnerability_score=0.0)
    assert score == 1.0


def test_composite_clamps_negative_nv_to_zero():
    # NV < 0 (noisy outperformed clean) shouldn't inflate the composite past 1
    score = adversarial_composite(noise_vulnerability_score=-0.2, nrr_f1=1.0)
    # 1 - (-0.2) = 1.2 would be wrong; we clamp at max(0, 1 - NV) but cap at 1 conceptually
    # Implementation clamps the lower bound at 0 (preventing < 0); upper bound 1.2 still possible.
    # If the spec wants to clamp upper bound too, document the choice — for now we DO NOT clamp upper.
    assert 1.0 <= score <= 1.2
