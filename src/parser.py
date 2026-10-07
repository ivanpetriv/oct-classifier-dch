import argparse
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="OCT Image Classification Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global arguments
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/run.yaml"),
        help="Path to the YAML configuration file.",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # train-final: Train on 100% of dataset
    train_final_parser = subparsers.add_parser(
        "train-final",
        help="Train the final model on 100%% of the dataset.",
    )
    train_final_parser.add_argument(
        "--train-path",
        type=Path,
        required=True,
        help="Path to the training dataset directory.",
    )
    train_final_parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_output"),
        help="Output directory for the trained model (default: model_output).",
    )

    # train-kfold: K-fold cross validation
    kfold_parser = subparsers.add_parser(
        "train-kfold",
        help="Train using Stratified K-Fold cross validation.",
    )
    kfold_parser.add_argument(
        "--train-path",
        type=Path,
        required=True,
        help="Path to the training dataset directory.",
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
        help="Run inference on a single image.",
    )
    inference_parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image to classify.",
    )
    inference_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the saved model file.",
    )

    # test: Evaluate on test set
    test_parser = subparsers.add_parser(
        "test",
        help="Evaluate a trained model on a hold-out test dataset.",
    )
    test_parser.add_argument(
        "--test-path",
        type=Path,
        required=True,
        help="Path to the test dataset directory.",
    )
    test_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the saved model file.",
    )

    # tune: Hyperparameter tuning
    tune_parser = subparsers.add_parser(
        "tune",
        help="Tune hyperparameters using Optuna.",
    )
    tune_parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset directory for tuning.",
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
        default=None,
        help="Number of Optuna trials to run.",
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
        type=str,
        default="optuna_tuning.db",
        help="Path to the Optuna SQLite database.",
    )
    tune_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tuning.",
    )

    # preprocess: Preprocess entire dataset
    preprocess_parser = subparsers.add_parser(
        "preprocess-dataset",
        help="Preprocess all images in the dataset and save to disk.",
    )
    preprocess_parser.add_argument(
        "--dataset-path",
        type=Path,
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
        help="Visualize feature extraction overlays on an image.",
    )
    viz_parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image to visualize.",
    )
    viz_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the visualization.",
    )

    # visualize-preprocessing: Preprocessing visualization
    preprocess_viz_parser = subparsers.add_parser(
        "visualize-preprocessing",
        help="Visualize the effect of each preprocessing step.",
    )
    preprocess_viz_parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image to visualize.",
    )
    preprocess_viz_parser.add_argument(
        "--output",
        type=Path,
        default=None,
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
        default=Path("config/run_schema.json"),
        help="Output path for the schema file (default: config/run_schema.json).",
    )

    return parser
