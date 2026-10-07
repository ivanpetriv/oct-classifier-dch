"""
Utility classes for Optuna.

This file contains functions that output info related to Optuna and its trials.
"""

import numpy as np
import optuna


def print_results(study: optuna.Study, top: int = 10) -> None:
    """
    Prints the results of the Optuna study
    """
    try:
        best_trial = study.best_trial
    except ValueError:
        print("No completed trials found.\n")
        return

    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    sorted_trials = sorted(
        completed_trials,
        key=lambda t: t.value if t.value is not None else -np.inf,
        reverse=True,
    )
    trial_values = [t.value for t in completed_trials if t.value is not None]

    # General statistics
    params_str = "\n    ".join(
        f"{k:<35} = {v}" for k, v in sorted(best_trial.params.items())
    )
    top_trials_str = "\n".join(
        f"{rank:<5} | #{trial.number:<6} | {trial.value:<10.6f}"
        for rank, trial in enumerate(sorted_trials[:top], start=1)
    )

    print(f"""
{"-" * 85}
Best Trial    : #{best_trial.number}
Best Macro F1 : {best_trial.value:.4f}

Best parameters:
    {params_str}

Total trials     : {len(study.trials)}
Completed trials : {len(completed_trials)}
Mean F1          : {np.mean(trial_values):.6f} (± {np.std(trial_values):.6f})
F1 Range         : {np.min(trial_values):.6f} to {np.max(trial_values):.6f}

Top {top} Trials:
{"Rank":<5} | {"Trial":<7} | {"F1 Score":<10}
{"-" * 35}
{top_trials_str}
{"-" * 85}
""")


def get_studies(db_path: str) -> list[optuna.study.StudySummary]:
    return optuna.study.get_all_study_summaries(storage=f"sqlite:///{db_path}")


def explore_studies(db_path: str) -> None:
    """List every Optuna study in the database."""
    studies = get_studies(db_path)

    if not studies:
        print(f"\nNo studies found in '{db_path}'.\n")
        return

    print(f"""
OPTUNA STUDIES IN '{db_path}'
{"Study Name":<35} | {"Trials":<8} | {"Direction":<10} | {"Best F1":<10} | {"Best Trial"}
{"-" * 85}""")

    for summary in studies:
        best_value = f"{summary.best_trial.value:.6f}" if summary.best_trial else "N/A"
        best_number = f"#{summary.best_trial.number}" if summary.best_trial else "N/A"

        print(
            f"{summary.study_name:<35} | "
            f"{summary.n_trials:<8} | "
            f"{summary.direction.name:<10} | "
            f"{best_value:<10} | "
            f"{best_number}"
        )
    print("-" * 85)


def explore_trials(db_path: str, study_name: str) -> None:
    """Print every trial in a specific study."""
    storage_url = f"sqlite:///{db_path}"

    try:
        study = optuna.load_study(storage=storage_url, study_name=study_name)
    except KeyError:
        print(f"\n[ERROR] Study '{study_name}' does not exist in '{db_path}'.\n")
        explore_studies(db_path)
        return

    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    failed = sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)

    print(f"""
TRIALS FOR STUDY: {study.study_name}
Total: {len(study.trials)}  |  Completed: {completed}  |  Pruned: {pruned}  |  Failed: {failed}
{"-" * 85}
""")

    sorted_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else -np.inf,
        reverse=True,
    )

    for trial in sorted_trials:
        value = f"{trial.value:.6f}" if trial.value is not None else "N/A"
        print(f"Trial #{trial.number:<5} | State: {trial.state.name:<10} | F1: {value}")

        if trial.params:
            for key, val in sorted(trial.params.items()):
                print(f"  ↳ {key:<30} = {val}")
        print()
    print("-" * 85 + "\n")


def analyze_results(db_path: str, study_name: str) -> None:
    storage_url = f"sqlite:///{db_path}"

    try:
        study = optuna.load_study(storage=storage_url, study_name=study_name)
    except KeyError:
        print(f"""
[ERROR] The study '{study_name}' does not exist in '{db_path}'.
Available studies in this database:
{[f"• {name}" for name in optuna.study.get_all_study_names(storage=storage_url)]}
""")
        return

    print_results(study)
