"""Structural checks for final-approach scripts and stored artifacts."""

from __future__ import annotations

from pathlib import Path


FINAL_APPROACH_DIR = Path(__file__).resolve().parents[1]
EXPECTED_GROUPS = {
    "logs/retained_policy_training": {
        "optuna12_100ksteps_buffer100k",
        "optuna12_600ksteps_buffer600k",
    },
    "logs/development_experiments": {
        "com_relative_states_first_input",
        "curriculum_experiment",
        "near_target_initialization",
        "observation_comparison",
        "optuna",
        "optuna12_200ksteps_buffer200k",
        "optuna12_300ksteps_buffer100k",
        "optuna12_300ksteps_buffer300k",
        "optuna12_600ksteps_buffer100k",
        "reward_shaping_trials",
    },
    "models/retained_policies": {
        "optuna12_100ksteps_buffer100k",
        "optuna12_600ksteps_buffer600k",
    },
    "models/development_experiments": {
        "com_relative_states_first_input",
        "curriculum_experiment",
        "near_target_initialization",
        "observation_comparison",
        "optuna12_200ksteps_buffer200k",
        "optuna12_300ksteps_buffer100k",
        "optuna12_300ksteps_buffer300k",
        "optuna12_600ksteps_buffer100k",
        "reward_shaping_trials",
    },
    "results/retained_policy_evaluations": {
        "eval_lqr_phase_50",
        "eval_many",
        "eval_multiseed",
        "optuna12_100ksteps_buffer100k",
        "optuna12_600ksteps_buffer600k",
        "policy_selection",
        "reconstructed_selected_policies_50rollouts",
        "reconstructed_supplemental_policies_50rollouts",
        "terminal_state_convergence",
    },
    "results/robustness_evaluations": {
        "robustness_expanded_test_bounds",
        "robustness_slight_ood",
        "robustness_larger_ood",
        "robustness_further_ood",
    },
    "results/retained_figures": {
        "episode_duration",
        "selected_policy_analysis",
        "training_curves",
    },
    "results/development_experiments": {
        "com_relative_states_first_input",
        "curriculum_experiment",
        "episode_return_learning_curves",
        "near_target_initialization",
        "observation_comparison",
        "optuna",
        "optuna12_200ksteps_buffer200k",
        "optuna12_300ksteps_buffer100k",
        "optuna12_300ksteps_buffer300k",
        "optuna12_600ksteps_buffer100k",
        "reward_shaping_trials",
        "round3_seed19_failure_audit",
    },
}


def test_root_scripts_follow_role_naming():
    allowed_support_modules = {"observation_space_env.py"}
    unexpected = {
        path.name
        for path in FINAL_APPROACH_DIR.glob("*.py")
        if not path.name.startswith(("evaluate_", "train_", "tune_"))
        and path.name not in allowed_support_modules
    }

    assert unexpected == set()


def test_plotters_contain_only_plot_scripts():
    plotters_dir = FINAL_APPROACH_DIR / "plotters"

    assert plotters_dir.is_dir()
    assert list(plotters_dir.glob("plot_*.py"))
    assert all(path.name.startswith("plot_") for path in plotters_dir.glob("*.py"))


def test_artifact_groups_are_in_approved_categories():
    for relative_category, expected_groups in EXPECTED_GROUPS.items():
        category = FINAL_APPROACH_DIR / relative_category
        actual_groups = {path.name for path in category.iterdir() if path.is_dir()}
        assert actual_groups == expected_groups


def test_retained_model_pairs_exist():
    retained_models = FINAL_APPROACH_DIR / "models" / "retained_policies"
    stems = {
        "optuna12_100ksteps_buffer100k":
            "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12",
        "optuna12_600ksteps_buffer600k":
            "sac_final_approach_portobs_rewardv9_600ksteps_seed11_optuna12_buffer600k",
    }

    for campaign, stem in stems.items():
        directory = retained_models / campaign
        assert (directory / f"{stem}.zip").is_file()
        assert (directory / f"{stem}_vecnormalize.pkl").is_file()


def test_readme_documents_retained_policy_pairs_and_reproduction():
    readme = (FINAL_APPROACH_DIR / "README.md").read_text(encoding="utf-8")

    assert "seed-12 100k" in readme
    assert "seed-11 600k" in readme
    assert "VecNormalize" in readme
    assert "evaluate_retained_sac_on_lqr_seeds_0_49.py" in readme
    assert "development_experiments" in readme
