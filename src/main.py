"""
Entry point.

Parses the CLI command and runs a corresponding function in functions.py.
"""

import sys
from pathlib import Path

from config.oct_config import OctConfig
from dataset.oct_dataset import DataLoader
from functions import (
    run_config_schema_generation,
    run_dataset_preprocessing,
    run_feature_extracting_visualization,
    run_inference,
    run_kfold,
    run_preprocessing_visualization,
    run_study,
    run_test,
    run_training,
)
from parser import get_parser


def main():
    """Main entry point for the OCT Classification Pipeline CLI."""
    parser = get_parser()
    args = parser.parse_args()

    # Load configuration
    try:
        config = OctConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        match args.command:
            case "train-final":
                if not args.train_path:
                    print("Error: Train dataset path required (--train-path or config)")
                    sys.exit(1)
                run_training(config, args.train_path, args.output)
                print(f"\n✓ Model training complete. Saved to: {args.output}")

            case "train-kfold":
                if not args.train_path:
                    print("Error: Train dataset path required (--train-path or config)")
                    sys.exit(1)
                run_kfold(
                    config,
                    args.train_path,
                    args.n_splits,
                    args.n_folds_to_use,
                )
                print("\n✓ K-Fold cross-validation complete")

            case "inference":
                if not args.image.exists():
                    print(f"Error: Image file not found: {args.image}")
                    sys.exit(1)
                if not args.model.exists():
                    print(f"Error: Model file not found: {args.model}")
                    sys.exit(1)
                run_inference(config, args.image, args.model)

            case "test":
                if not args.test_path:
                    print("Error: Test dataset path required (--test-path or config)")
                    sys.exit(1)
                if not args.model.exists():
                    print(f"Error: Model file not found: {args.model}")
                    sys.exit(1)
                run_test(config, Path(args.test_path), args.model)
                print("\n✓ Model evaluation complete")

            case "tune":
                print(
                    f"Running optuna tuning (study: {args.study_name}, db: {args.db})"
                )
                print(
                    f"Trials: {args.n_trials}, Jobs: {args.jobs}, Splits: {args.n_splits}"
                )

                run_study(
                    config=config,
                    dataset_path=args.dataset_path,
                    db_path=args.db,
                    study_name=args.study_name,
                    n_trials=args.n_trials or 10,
                    n_splits=args.n_splits,
                    n_folds=args.n_folds_to_use,
                    n_jobs=args.jobs,
                    random_state=config.random_state,
                )

            case "preprocess-dataset":
                if not args.train_path:
                    print("Error: Train dataset path required (--train-path or config)")
                    sys.exit(1)
                if not args.output:
                    print("Error: Output path required (--output or config)")
                    sys.exit(1)
                run_dataset_preprocessing(config, Path(args.output))
                print(f"\n✓ Dataset preprocessing complete. Saved to: {args.output}")

            case "visualize-features":
                if not args.image.exists():
                    print(f"Error: Image file not found: {args.image}")
                    sys.exit(1)
                run_feature_extracting_visualization(config, args.image, args.output)
                if args.output:
                    print(f"\n✓ Feature visualization saved to: {args.output}")
                else:
                    print("\n✓ Feature visualization displayed")

            case "visualize-preprocessing":
                if not args.image.exists():
                    print(f"Error: Image file not found: {args.image}")
                    sys.exit(1)
                run_preprocessing_visualization(config, args.image, args.output)
                if args.output:
                    print(f"\n✓ Preprocessing visualization saved to: {args.output}")
                else:
                    print("\n✓ Preprocessing visualization displayed")

            case "generate-schema":
                run_config_schema_generation(args.output)
                print(f"\n✓ Schema generated: {args.output}")

            case _:
                parser.print_help()
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
