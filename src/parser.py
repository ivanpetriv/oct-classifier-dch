import argparse
import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()


def existing_path(path_str: str) -> Path:
    """Validates that a file or directory exists and returns a Path."""
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path not found: '{path}'")
    return path


def env_file(folder: str | None, filename: str) -> str | None:
    """Joins an environment folder string with a default filename."""
    return str(Path(folder) / filename) if folder else None


def get_parser() -> argparse.ArgumentParser:
    """Creates and configures the CLI argument parser."""

    TRAIN_DATASET_PATH = os.getenv("TRAIN_DATASET_PATH")
    TEST_DATASET_PATH = os.getenv("TEST_DATASET_PATH")
    MODEL_SAVE_FOLDER = os.getenv("MODEL_SAVE_FOLDER")
    RUN_DATA_SAVE_FOLDER = os.getenv("RUN_DATA_SAVE_FOLDER")
    OPTUNA_DB_FOLDER = os.getenv("OPTUNA_DB_FOLDER")
    CONFIG_FOLDER = os.getenv("CONFIG_FOLDER", "config")
    default_config_path = env_file(CONFIG_FOLDER, "run.yaml")
    default_model_path = env_file(MODEL_SAVE_FOLDER, "model.joblib")
    default_run_data_path = env_file(RUN_DATA_SAVE_FOLDER, "run_data.json")
    default_optuna_db_path = env_file(OPTUNA_DB_FOLDER, "optuna.db")
    default_schema_path = env_file(CONFIG_FOLDER, "run_schema.json")

    parser = argparse.ArgumentParser(
        description="OCT Image Classification Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument(
        "--config",
        type=existing_path,
        default=default_config_path,
        required=default_config_path is None,
        help="Path to the YAML configuration file (default: CONFIG_FOLDER/run.yaml).",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # train-final: Train on 100% of the dataset
    train_final_parser = subparsers.add_parser(
        "train-final",
        parents=[config_parent],
        help="Train the final model on 100%% of the dataset.",
    )
    train_final_parser.add_argument(
        "--train-data",
        type=existing_path,
        default=TRAIN_DATASET_PATH,
        required=TRAIN_DATASET_PATH is None,
        help="Path to the training dataset directory (default: TRAIN_DATASET_PATH).",
    )
    train_final_parser.add_argument(
        "--save-model",
        type=Path,
        default=MODEL_SAVE_FOLDER,
        required=MODEL_SAVE_FOLDER is None,
        help="Output directory for the trained model (default: MODEL_SAVE_FOLDER).",
    )
    train_final_parser.add_argument(
        "--save-run-data",
        type=Path,
        default=RUN_DATA_SAVE_FOLDER,
        required=RUN_DATA_SAVE_FOLDER is None,
        help="Output directory for run data (default: RUN_DATA_SAVE_FOLDER).",
    )

    # train-kfold: K-fold cross validation
    kfold_parser = subparsers.add_parser(
        "train-kfold",
        parents=[config_parent],
        help="Train using Stratified K-Fold cross validation.",
    )
    kfold_parser.add_argument(
        "--train-data",
        type=existing_path,
        default=TRAIN_DATASET_PATH,
        required=TRAIN_DATASET_PATH is None,
        help="Path to the training dataset directory (default: TRAIN_DATASET_PATH).",
    )
    kfold_parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    kfold_parser.add_argument(
        "--n-folds-to-use",
        type=int,
        default=None,
        help="Number of folds to actually train (default: all).",
    )

    # inference: Single image inference
    inference_parser = subparsers.add_parser(
        "inference",
        parents=[config_parent],
        help="Run inference on a single image.",
    )
    inference_parser.add_argument(
        "--image",
        type=existing_path,
        required=True,
        help="Path to the image to classify.",
    )
    inference_parser.add_argument(
        "--model",
        type=existing_path,
        default=default_model_path,
        required=default_model_path is None,
        help="Path to the saved model file (default: MODEL_SAVE_FOLDER/model.joblib).",
    )

    # test: Evaluate on test set
    test_parser = subparsers.add_parser(
        "test",
        parents=[config_parent],
        help="Evaluate a trained model on a hold-out test dataset.",
    )
    test_parser.add_argument(
        "--test-data",
        type=existing_path,
        default=TEST_DATASET_PATH,
        required=TEST_DATASET_PATH is None,
        help="Path to the test dataset directory (default: TEST_DATASET_PATH).",
    )
    test_parser.add_argument(
        "--model",
        type=existing_path,
        default=default_model_path,
        required=default_model_path is None,
        help="Path to the saved model file (default: MODEL_SAVE_FOLDER/model.joblib).",
    )

    # tune: Hyperparameter tuning
    tune_parser = subparsers.add_parser(
        "tune",
        parents=[config_parent],
        help="Tune hyperparameters using Optuna.",
    )
    tune_parser.add_argument(
        "--dataset",
        type=existing_path,
        default=TRAIN_DATASET_PATH,
        required=TRAIN_DATASET_PATH is None,
        help="Path to the dataset directory for tuning (default: TRAIN_DATASET_PATH).",
    )
    tune_parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    tune_parser.add_argument(
        "--n-folds-to-use",
        type=int,
        default=None,
        help="Number of folds to actually train (default: all).",
    )
    tune_parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of Optuna trials to run (default: 5).",
    )
    tune_parser.add_argument(
        "--study-name",
        type=str,
        default="oct_tuning",
        help="Optuna study name.",
    )
    tune_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel Optuna jobs.",
    )
    tune_parser.add_argument(
        "--db",
        type=Path,
        default=default_optuna_db_path,
        help="Path to the Optuna SQLite database (default: RUN_DATA_SAVE_FOLDER/optuna_tuning.db).",
    )

    # preprocess: Preprocess entire dataset
    preprocess_parser = subparsers.add_parser(
        "preprocess-dataset",
        parents=[config_parent],
        help="Preprocess all images in the dataset and save to disk.",
    )
    preprocess_parser.add_argument(
        "--data",
        type=existing_path,
        required=True,
        help="Path to the raw dataset to preprocess.",
    )
    preprocess_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for preprocessed images.",
    )

    # visualize: Feature extraction visualization
    viz_parser = subparsers.add_parser(
        "visualize-features",
        parents=[config_parent],
        help="Visualize feature extraction overlays on an image.",
    )
    viz_parser.add_argument(
        "--image",
        type=existing_path,
        required=True,
        help="Path to the image to visualize.",
    )
    viz_parser.add_argument(
        "--output-image",
        type=Path,
        default=RUN_DATA_SAVE_FOLDER,
        help="Optional path to save the visualization.",
    )

    # visualize-preprocessing: Preprocessing visualization
    preprocess_viz_parser = subparsers.add_parser(
        "visualize-preprocessing",
        parents=[config_parent],
        help="Visualize the effect of each preprocessing step.",
    )
    preprocess_viz_parser.add_argument(
        "--image",
        type=existing_path,
        required=True,
        help="Path to the image to visualize.",
    )
    preprocess_viz_parser.add_argument(
        "--output-image",
        type=Path,
        default=RUN_DATA_SAVE_FOLDER,
        help="Optional path to save the visualization.",
    )

    # schema: Generate JSON schema
    schema_parser = subparsers.add_parser(
        "generate-schema",
        help="Generate JSON schema for the YAML configuration file.",
    )
    schema_parser.add_argument(
        "--output",
        type=Path,
        default=default_schema_path,
        required=default_schema_path is None,
        help="Output path for the schema file (default: CONFIG_FOLDER/run_schema.json).",
    )

    return parser
