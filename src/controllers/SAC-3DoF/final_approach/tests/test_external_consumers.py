"""Checks for consumers of relocated final-approach artifacts."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
FINAL_APPROACH_DIR = Path(__file__).resolve().parents[1]
CONSUMERS = (
    REPO_ROOT / "src/controllers/run_mixed_full_mission_evaluations.py",
    REPO_ROOT / "src/controllers/plotters/analyze_final_approach_control_effort.py",
    REPO_ROOT / "src/controllers/plotters/plot_terminal_state_convergence.py",
    REPO_ROOT / "src/controllers/SAC-3DoF/plotters/plot_sac_evaluation_summary.py",
    REPO_ROOT / "src/controllers/cbf_3DoF/tune_final_approach_cbf.py",
    REPO_ROOT / "src/controllers/cbf_3DoF/evaluate_final_approach_cbf.py",
    REPO_ROOT / "src/controllers/cbf_3DoF/plot_sac_cbf_trajectory_comparisons.py",
)


def test_external_consumers_do_not_use_flat_artifact_paths():
    forbidden = (
        '"models" / "optuna12_',
        '"results" / "eval_',
        "final_approach/results/eval_",
        "final_approach/results/reconstructed_",
    )

    offenders = {
        path.name: fragment
        for path in CONSUMERS
        for fragment in forbidden
        if fragment in path.read_text(encoding="utf-8")
    }

    assert offenders == {}


def test_final_approach_consumers_use_explicit_horizons():
    offenders = [
        path.name
        for path in CONSUMERS
        if "max_time=None" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_retained_external_inputs_exist():
    retained_models = FINAL_APPROACH_DIR / "models" / "retained_policies"
    retained_results = (
        FINAL_APPROACH_DIR / "results" / "retained_policy_evaluations"
    )

    assert (
        retained_models / "optuna12_100ksteps_buffer100k"
        / "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12.zip"
    ).is_file()
    assert (
        retained_models / "optuna12_600ksteps_buffer600k"
        / "sac_final_approach_portobs_rewardv9_600ksteps_seed11_optuna12_buffer600k.zip"
    ).is_file()
    assert (
        retained_results / "reconstructed_selected_policies_50rollouts"
        / "all_rollouts.csv"
    ).is_file()
