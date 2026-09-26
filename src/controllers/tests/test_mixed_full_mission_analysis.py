import numpy as np
import pytest

from controllers.analyze_mixed_full_mission_results import stats, wilson


def test_empty_statistics_have_an_explicit_representation():
    assert stats(np.array([], dtype=float)) == {
        "n": 0,
        "mean": None,
        "sample_std": None,
        "min": None,
        "max": None,
    }


def test_wilson_interval_requires_a_nonempty_sample():
    with pytest.raises(ValueError, match="positive"):
        wilson(0, 0)
