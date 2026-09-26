import numpy as np
import pytest

from controllers.plotters.plot_terminal_state_convergence import (
    representative_index,
    validate,
)


def test_cache_arrays_must_match_the_seed_count():
    data = {
        "seeds": np.array([0, 1]),
        "times": np.ones((2, 3)),
        "success": np.ones(2, dtype=bool),
        "error": np.ones((1, 3)),
    }

    with pytest.raises(ValueError, match="error"):
        validate(data, ("seeds", "times", "success", "error"))


def test_representative_rollout_requires_at_least_one_finite_time():
    times = np.full((2, 3), np.nan)

    with pytest.raises(ValueError, match="finite"):
        representative_index(times, np.array([True, False]))
