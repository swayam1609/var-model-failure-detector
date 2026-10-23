
import numpy as np
import pytest

from src.risk_models import (
    historical_var,
    gaussian_var,
    student_t_var,
)


def test_historical_var_known_distribution():
    returns = np.array([-0.10, -0.05, 0.01, 0.02])
    result = historical_var(returns, confidence=0.75)

    assert result == pytest.approx(0.10)


def test_gaussian_var_is_finite():
    returns = np.array([-0.03, -0.02, -0.01, 0.01, 0.02, 0.03])
    result = gaussian_var(returns, confidence=0.95)

    assert np.isfinite(result)


def test_student_t_var_is_finite():
    rng = np.random.default_rng(42)
    returns = rng.standard_t(df=5, size=1000) * 0.01

    result = student_t_var(returns, confidence=0.95)

    assert np.isfinite(result)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1])
def test_invalid_confidence_is_rejected(confidence):
    returns = np.array([-0.02, -0.01, 0.01, 0.02])

    with pytest.raises(ValueError):
        historical_var(returns, confidence)


def test_student_t_requires_enough_observations():
    returns = np.array([-0.03, -0.02, -0.01, 0.01, 0.02])

    with pytest.raises(ValueError):
        student_t_var(returns, confidence=0.95)


def test_non_finite_returns_are_ignored():
    returns = np.array([-0.02, -0.01, 0.01, 0.02, np.nan])

    result = historical_var(returns, confidence=0.75)

    assert np.isfinite(result)