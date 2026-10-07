"""
Start points for code execution.

When main.py has parsed the command, it runs one of these functions.
"""

import json
import os
from pathlib import Path
from typing import Any

import joblib  # type: ignore
import numpy as np
from PIL import Image
import optuna
from sklearn.metrics import classification_report  # type: ignore

from config.oct_config import OctConfig
from dataset.oct_dataset import DataLoader
from pipeline._pipeline_builders.feature_extraction_builder import (
    create_feature_extractors_from_config,
)
from pipeline._pipeline_builders.preprocessing_builder import (
    create_preprocessors_from_config,
)
from pipeline.pipeline_builder import build_pipeline
from pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from pipeline.validation.cross_validation import CrossValidation
from services.save_run_data import save_run_data
from tuning.converter import generate_optuna_seed
from tuning.early_stopping_callback import EarlyStoppingCallback
from tuning.objective import OptunaTuner
from tuning.utils import create_study
from visualization.dch_feature_extraction import FeatureExtractorVisualizer


def run_training(
    config: OctConfig, train_dataset_path: Path, new_model_path: Path | None
):
    """
    Trains the final model on 100% of the dataset and saves it to the output path.
    """
    if config.preprocessing is not None:
        preprocessing = PreprocessingPipeline(
            create_preprocessors_from_config(config.preprocessing)
        )
    else:
        preprocessing = None

    train_dataset = DataLoader(
        dataset_path=train_dataset_path,
        allowed_classes=config.dataset.classes,
        samples_per_class=config.dataset.max_samples_per_class,
        random_state=config.random_state,
        preprocessor=preprocessing,
    )

    pipeline = build_pipeline(config)

    all_indices = np.arange(len(train_dataset.labels))

    X_all, y_all = train_dataset.get_batch(
        all_indices, description="Full Dataset Train"
    )

    pipeline.fit(X_all, y_all)

    if new_model_path is not None:
        y_pred = pipeline.predict(X_all)
        save_run_data(new_model_path, config, y_all, y_pred)

        path_obj = Path(new_model_path / "model.joblib")
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, str(path_obj))


def run_kfold(
    config: OctConfig,
    train_dataset_path: Path,
    n_splits: int,
    n_folds_to_use: int | None,
):
    """
    Runs a k-fold validation on the dataset to measure the average F1-score across all folds.
    """
    if n_folds_to_use is None:
        n_folds_to_use = n_splits

    if config.preprocessing is not None:
        preprocessing = PreprocessingPipeline(
            create_preprocessors_from_config(config.preprocessing)
        )
    else:
        preprocessing = None

    dataset = DataLoader(
        dataset_path=train_dataset_path,
        allowed_classes=config.dataset.classes,
        samples_per_class=config.dataset.max_samples_per_class,
        preprocessor=preprocessing,
        random_state=config.random_state,
    )

    pipeline = build_pipeline(config)

    runner = CrossValidation(pipeline, dataset, n_splits, config.random_state)

    runner.run(n_folds_to_use)


def run_inference(config: OctConfig, image_path: Path, model_path: Path):
    """
    Classifies a given image into a class.
    """
    with Image.open(image_path) as img:
        img = img.convert("L")
        image = np.array(img)

    if config.preprocessing is not None:
        preprocessing = PreprocessingPipeline(
            create_preprocessors_from_config(config.preprocessing)
        )
        image = preprocessing.execute(image)

    model = joblib.load(model_path)

    idx_to_class = {v: k for k, v in config.dataset.classes.items()}

    prediction_idx = model.predict([image])[0]
    predicted_class = idx_to_class[int(prediction_idx)]

    print(predicted_class)


def run_test(config: OctConfig, test_dataset_path: Path, model_path: Path):
    """
    Runs a test to measure final results.
    """
    if config.preprocessing is not None:
        preprocessing = PreprocessingPipeline(
            create_preprocessors_from_config(config.preprocessing)
        )
    else:
        preprocessing = None

    model = joblib.load(model_path)

    test_dataset = DataLoader(
        dataset_path=test_dataset_path,
        allowed_classes=config.dataset.classes,
        samples_per_class=config.dataset.max_samples_per_class,
        random_state=config.random_state,
        preprocessor=preprocessing,
    )

    total_samples = len(test_dataset.labels)
    if total_samples == 0:
        raise ValueError("Test dataset is empty or paths are incorrect.")

    all_indices = np.arange(total_samples)
    X_test, y_test = test_dataset.get_batch(
        all_indices, description="Loading Test Data"
    )

    test_preds = model.predict(X_test)

    print("\n==================== TEST SET RESULTS ====================")
    print(classification_report(y_test, test_preds, zero_division=0))

    # TODO: Add cis
    # TODO: add visualization


def run_study(
    *,
    config: OctConfig,
    dataset_path: Path,
    db_path: str,
    study_name: str,
    n_trials: int,
    n_splits: int,
    n_folds: int | None,
    n_jobs: int,
    random_state: int | None,
) -> optuna.Study:

    print(f"Tuning the study:                  {study_name}")
    print("=" * 70)

    tuner = OptunaTuner(
        base_config=config,
        dataset_path=dataset_path,
        n_folds=n_splits,
        n_folds_to_use=n_folds,
        random_state=random_state,
    )

    study = create_study(
        db_path=db_path,
        study_name=study_name,
        seed=random_state,
    )

    if len(study.trials) == 0:
        start_values = generate_optuna_seed(config)

        if start_values:
            print("\nEnqueuing baseline config:")
            print(json.dumps(start_values, indent=4))
            study.enqueue_trial(start_values)

    study.optimize(
        tuner.objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=[EarlyStoppingCallback(patience=100)],
    )

    os.system("paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null")

    return study


def run_dataset_preprocessing(config: OctConfig, preprocessed_save_path: Path):
    """
    Runs image preprocessing on the dataset and saves the preprocessed images on disc.
    """


def run_dataset_pruning(config: OctConfig):
    """
    Deletes images from the dataset that don't correspond to the criteria.
    """


def run_feature_extracting_visualization(
    config: OctConfig, image_path: Path, save_vis_path: Path
):
    """
    Visualizes the lines across which the feature extraction happens.
    """
    if not config.feature_extraction:
        raise ValueError("No feature extractors configured in run.yaml")

    extractors = create_feature_extractors_from_config(config.feature_extraction)

    with Image.open(image_path) as file_object:
        img_arr = np.array(file_object.convert("L"))
        visualizer = FeatureExtractorVisualizer(extractors)
        visualizer.visualize(img_arr)


def run_preprocessing_visualization(
    config: OctConfig, image_path: Path, save_vis_path: Path
):
    """
    Visualizes the effect of each preprocessing step on an image.
    """


def run_config_schema_generation(output_file: Path):
    """
    Creates a schema file for YAML run files.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    schema = OctConfig.model_json_schema()

    with open(output_file, "w") as fp:
        json.dump(schema, fp, indent=4, default=str)

    print(f"Schema successfully generated at {output_file}")
