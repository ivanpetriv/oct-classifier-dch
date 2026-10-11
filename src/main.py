"""
Entry point.

Parses the CLI command and runs a corresponding function in functions.py.
"""

import sys

from config.oct_config import OctConfig
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
    """Main entry point."""
    parser = get_parser()
    args = parser.parse_args()

    if args.command == "generate-schema":
        run_config_schema_generation(args.output)
        print(f"\n✓ Schema generated: {args.output}")
        sys.exit(0)

    try:
        config = OctConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        match args.command:
            case "train-final":
                if not args.train_data:
                    print(
                        "Error: Train dataset path required (--train-data or TRAIN_DATASET_PATH in .env)"
                    )
                    sys.exit(1)
                if not args.save_model:
                    print(
                        "Error: Model save path required (--save-model or MODEL_SAVE_FOLDER in .env)"
                    )
                    sys.exit(1)
                # Pass args.save_run_data as well if run_training expects it
                run_training(config, args.train_data, args.save_model)
                print(f"\n✓ Model training complete. Saved to: {args.save_model}")

            case "train-kfold":
                if not args.train_data:
                    print(
                        "Error: Train dataset path required (--train-data or TRAIN_DATASET_PATH in .env)"
                    )
                    sys.exit(1)
                run_kfold(
                    config,
                    args.train_data,
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
                if not args.test_data:
                    print(
                        "Error: Test dataset path required (--test-data or TEST_DATASET_PATH in .env)"
                    )
                    sys.exit(1)
                if not args.model.exists():
                    print(f"Error: Model file not found: {args.model}")
                    sys.exit(1)
                run_test(config, args.test_data, args.model)
                print("\n✓ Model evaluation complete")

            case "tune":
                if not args.dataset:
                    print(
                        "Error: Dataset path required (--dataset or TRAIN_DATASET_PATH in .env)"
                    )
                    sys.exit(1)
                print(
                    f"Running optuna tuning (study: {args.study_name}, db: {args.db})"
                )
                print(
                    f"Trials: {args.n_trials}, Jobs: {args.jobs}, Splits: {args.n_splits}"
                )

                run_study(
                    config=config,
                    dataset_path=args.dataset,
                    db_path=args.db,
                    study_name=args.study_name,
                    n_trials=args.n_trials or 10,
                    n_splits=args.n_splits,
                    n_folds=args.n_folds_to_use,
                    n_jobs=args.jobs,
                    random_state=config.random_state,
                )

            case "preprocess-dataset":
                if not args.data.exists():
                    print(f"Error: Raw dataset path not found: {args.data}")
                    sys.exit(1)
                # Ensure run_dataset_preprocessing accepts (config, input_path, output_path)
                run_dataset_preprocessing(config, args.data, args.output)
                print(f"\n✓ Dataset preprocessing complete. Saved to: {args.output}")

            case "visualize-features":
                if not args.image.exists():
                    print(f"Error: Image file not found: {args.image}")
                    sys.exit(1)
                run_feature_extracting_visualization(
                    config, args.image, args.output_image
                )
                if args.output_image:
                    print(f"\n✓ Feature visualization saved to: {args.output_image}")
                else:
                    print("\n✓ Feature visualization displayed")

            case "visualize-preprocessing":
                if not args.image.exists():
                    print(f"Error: Image file not found: {args.image}")
                    sys.exit(1)
                run_preprocessing_visualization(config, args.image, args.output_image)
                if args.output_image:
                    print(
                        f"\n✓ Preprocessing visualization saved to: {args.output_image}"
                    )
                else:
                    print("\n✓ Preprocessing visualization displayed")

            case _:
                parser.print_help()
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()