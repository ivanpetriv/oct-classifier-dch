import copy
import os
from pathlib import Path

import optuna

from config.oct_config import OctConfig
from dataset.oct_dataset import DataLoader
from pipeline._pipeline_builders.preprocessing_builder import (
    create_preprocessors_from_config,
)
from pipeline.pipeline_builder import build_pipeline
from pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from pipeline.validation.cross_validation import CrossValidation
from tuning.tuner_builders import (
    create_classification_tuner,
    create_dimensionality_reduction_tuner,
    create_feature_extraction_tuner,
    create_preprocessing_tuner,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class OptunaTuner:
    """
    Tuner that attempts different configurations of hyperparameters to find the one
    which yields the highest F1-score.
    """

    def __init__(
        self,
        base_config: OctConfig,
        dataset_path: Path,
        n_folds: int,
        n_folds_to_use: int | None,
        random_state: int | None,
    ) -> None:
        self.base_config = base_config
        self.dataset_path = dataset_path
        self.n_folds = n_folds
        self.n_folds_to_use = n_folds_to_use
        self.random_state = random_state

    def objective(self, trial: optuna.Trial) -> float | None:
        """
        Runs the pipeline with the generated config.
        Args:
            trial (optuna.Trial): _description_
        Raises:
            optuna.TrialPruned: If an exception is thrown during the trial, it prunes the trial.
        Returns:
            float representing the F1-score of the trial.
        """
        config = self._configure_trial(self.base_config, trial)

        pipeline = build_pipeline(config)

        try:
            if config.preprocessing is not None:
                preprocessing = PreprocessingPipeline(
                    create_preprocessors_from_config(config.preprocessing)
                )
            else:
                preprocessing = None

            dataset = DataLoader(
                dataset_path=self.dataset_path,
                allowed_classes=config.dataset.classes,
                samples_per_class=config.dataset.max_samples_per_class,
                random_state=config.random_state,
                preprocessor=preprocessing,
            )

            runner = CrossValidation(
                pipeline,
                dataset,
                self.random_state,
            )

            return runner.run(
                self.n_folds,
                self.n_folds_to_use,
            )
        except ValueError as e:
            if "infs or NaNs" in str(e):
                trial.report(0.0, step=0)
                raise optuna.TrialPruned()

    def _configure_trial(
        self,
        base_config: OctConfig,
        trial: optuna.Trial,
    ) -> OctConfig:
        """
        Sets up the config for the trial, injecting the generated values by Optuna.
        Args:
            base_config (OctConfig): Base configuration that will be overridden.
            trial (optuna.Trial): Trial.
        Returns:
            OctConfig: New configuration with the updated values.
        """
        config = copy.deepcopy(base_config)

        if config.preprocessing is not None:
            config.preprocessing = create_preprocessing_tuner(
                config.preprocessing, trial
            )

        if config.feature_extraction is not None:
            config.feature_extraction = create_feature_extraction_tuner(
                config.feature_extraction, trial
            )

        if config.dimensionality_reduction is not None:
            config.dimensionality_reduction = create_dimensionality_reduction_tuner(
                config.dimensionality_reduction, trial
            )

        config.classification = create_classification_tuner(
            config.classification, trial
        )

        return config
