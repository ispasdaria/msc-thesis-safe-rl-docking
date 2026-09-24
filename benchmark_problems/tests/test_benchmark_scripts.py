from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
MOUNTAIN_CAR_DIR = BENCHMARK_DIR / "mountain_car_exercise"
PENDULUM_DIR = BENCHMARK_DIR / "pendulum_exercise"


def load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeModel:
    def predict(self, observation, deterministic=True):
        return np.array([observation[0]], dtype=np.float32), None


class FakeMountainCarEnv:
    def __init__(self):
        self._transitions = iter(
            [
                (np.array([1.0, 0.1]), 10.0, False, False, {}),
                (np.array([2.0, 0.2]), 20.0, True, False, {}),
            ]
        )
        self.closed = False

    def reset(self, seed=None):
        return np.array([0.0, 0.0]), {}

    def step(self, action):
        return next(self._transitions)

    def close(self):
        self.closed = True


class FakeRetreatingMountainCarEnv(FakeMountainCarEnv):
    def __init__(self):
        super().__init__()
        self._transitions = iter(
            [
                (np.array([1.0, -0.1]), -1.0, False, False, {}),
                (np.array([0.0, -0.2]), -2.0, True, False, {}),
            ]
        )

    def reset(self, seed=None):
        return np.array([2.0, 0.0]), {}


def test_single_mountain_car_rollout_records_resulting_states():
    module = load_script(MOUNTAIN_CAR_DIR / "evaluate_sac_mountain_car.py")

    history = module.run_episode(FakeMountainCarEnv(), FakeModel(), seed=1)

    np.testing.assert_allclose(history["positions"], [1.0, 2.0])
    np.testing.assert_allclose(history["velocities"], [0.1, 0.2])
    np.testing.assert_allclose(history["actions"], [0.0, 1.0])
    np.testing.assert_allclose(history["rewards"], [10.0, 20.0])


def test_seed_evaluation_uses_terminal_state(monkeypatch):
    module = load_script(
        MOUNTAIN_CAR_DIR / "train_and_evaluate_sac_mountain_car_seeds.py"
    )
    env = FakeMountainCarEnv()
    monkeypatch.setattr(module.SAC, "load", lambda _path: FakeModel())
    monkeypatch.setattr(module.gym, "make", lambda _env_id: env)

    result = module.evaluate_model("model.zip", seed=3, n_eval_episodes=1)[0]

    assert result["final_position"] == pytest.approx(2.0)
    assert result["final_velocity"] == pytest.approx(0.2)
    assert result["max_position"] == pytest.approx(2.0)
    assert env.closed


def test_seed_evaluation_includes_initial_state_in_maximum(monkeypatch):
    module = load_script(
        MOUNTAIN_CAR_DIR / "train_and_evaluate_sac_mountain_car_seeds.py"
    )
    env = FakeRetreatingMountainCarEnv()
    monkeypatch.setattr(module.SAC, "load", lambda _path: FakeModel())
    monkeypatch.setattr(module.gym, "make", lambda _env_id: env)

    result = module.evaluate_model("model.zip", seed=3, n_eval_episodes=1)[0]

    assert result["max_position"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "filename",
    [
        "evaluate_sac_mountain_car.py",
        "train_sac_mountain_car.py",
        "train_and_evaluate_sac_mountain_car_seeds.py",
        "watch_sac_mountain_car.py",
    ],
)
def test_mountain_car_scripts_define_file_relative_base_directory(filename):
    source = (MOUNTAIN_CAR_DIR / filename).read_text(encoding="utf-8")
    assert "Path(__file__).resolve().parent" in source


def test_mountain_car_scripts_have_no_windows_notification_dependencies():
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in MOUNTAIN_CAR_DIR.glob("*.py")
    )
    assert "winsound" not in sources
    assert "win32com" not in sources
    assert "notify_done" not in sources


@pytest.mark.parametrize(
    "filename",
    [
        "evaluate_sac_pendulum.py",
        "plot_sac_pendulum.py",
        "train_sac_pendulum.py",
        "train_sac_pendulum_tuned.py",
    ],
)
def test_pendulum_scripts_use_main_guard(filename):
    tree = ast.parse((PENDULUM_DIR / filename).read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)
    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    )


@pytest.mark.parametrize(
    "filename", ["train_sac_pendulum.py", "train_sac_pendulum_tuned.py"]
)
def test_pendulum_training_uses_fixed_seed(filename):
    source = (PENDULUM_DIR / filename).read_text(encoding="utf-8")
    assert "SEED = 0" in source
    assert "env.reset(seed=SEED)" in source
    assert "env.action_space.seed(SEED)" in source
    assert "seed=SEED" in source


def test_comments_are_objective_and_impersonal():
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in BENCHMARK_DIR.rglob("*.py")
        if "tests" not in path.parts
    )
    assert "Usually the best model" not in sources
    assert "somehow does not exist" not in sources
    assert "Your logging / display choices" not in sources
    assert "Main SAC defaults from Stable-Baselines3" not in sources
