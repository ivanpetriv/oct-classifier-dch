import optuna


def create_study(
    db_path: str,
    study_name: str,
    seed: int | None,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=1,
    )

    return optuna.create_study(
        storage=f"sqlite:///{db_path}",
        study_name=study_name,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
