## This is the refactored version of OCT classifier, and it's not complete.

The original version is here:
https://github.com/ivanpetriv/oct-classifier-v3.git


---

# Classification of Optical Coherence Tomography Images Using Distributed Cumulative Histograms and Machine Learning

A command-line interface for training, tuning, evaluating, and running inference with an OCT image classification model.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Commands](#commands)

## Installation

```bash
# Clone the repository
git clone https://github.com/ivanpetriv/oct-classifier-dch.git
cd oct-classifier-dch

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Build the Rust extension (requires a Rust toolchain: https://rustup.rs)
pip install maturin
maturin develop --release -m rust-src/Cargo.toml

# Install the Python packages
pip install -e .
```


## Configuration

Defaults for most paths come from environment variables. CLI arguments that use them are not required but have a higher precedence if provided. If a corresponding environment variable is not provided, the CLI argument becomes required.

### Environment variables

Variables are loaded from an `.env` file in the working directory.

| Variable | Used for |
|---|---|
| `TRAIN_DATASET_PATH` | Training dataset directory |
| `TEST_DATASET_PATH` | Test dataset directory |
| `MODEL_SAVE_FOLDER` | Where models are saved / loaded from |
| `RUN_DATA_SAVE_FOLDER` | Where run data and visualizations are saved |
| `OPTUNA_DB_FOLDER` | Folder holding the Optuna SQLite database |
| `CONFIG_FOLDER` | Folder containing the YAML config and schema |

Example `.env`:

```
TRAIN_DATASET_PATH = data/train
TEST_DATASET_PATH = data/test
MODEL_SAVE_FOLDER = artifacts/models
RUN_DATA_SAVE_FOLDER = artifacts/runs
OPTUNA_DB_FOLDER = artifacts/optuna
CONFIG_FOLDER = config
```


## Usage


After that, a command can be run like this:

```bash
python3 src/main.py <command> [options]
```

Show help for any command with `--help`:

```bash
python3 src/main.py --help
python3 src/main.py train-kfold --help
```

## Commands

All commands except `generate-schema` share this option:

| Option | Description | Default |
|---|---|---|
| `--config PATH` | YAML configuration file | `$CONFIG_FOLDER/run.yaml` |

### train-final

Train the final model on 100% of the dataset.

| Option | Description | Default |
|---|---|---|
| `--train-data` | Training dataset directory | `$TRAIN_DATASET_PATH` |
| `--save-model` | Output directory for the trained model | `$MODEL_SAVE_FOLDER` |
| `--save-run-data` | Output directory for the configuration | `$RUN_DATA_SAVE_FOLDER` |

### train-kfold

Train using Stratified K-Fold cross-validation.

| Option | Description | Default |
|---|---|---|
| `--train-data` | Training dataset directory | `$TRAIN_DATASET_PATH` |
| `--n-splits` | Number of cross-validation folds | `5` |
| `--n-folds-to-use` | Number of folds to actually train | all folds |


### tune

Tune hyperparameters with Optuna using cross-validation.

| Option | Description | Default |
|---|---|---|
| `--dataset` | Dataset directory for tuning | `$TRAIN_DATASET_PATH` |
| `--n-splits` | Number of cross-validation folds | `5` |
| `--n-folds-to-use` | Number of folds to actually train | all folds |
| `--n-trials` | Number of Optuna trials | `5` |
| `--study-name` | Optuna study name | `oct_tuning` |
| `--jobs` | Number of parallel Optuna jobs | `1` |
| `--db` | Optuna SQLite database path | `$OPTUNA_DB_FOLDER/optuna.db` |

Because results are stored in a SQLite database, reusing the same `--study-name` and `--db` lets you resume or extend a study.

### test

Evaluate a trained model on a hold-out test dataset.

| Option | Description | Default |
|---|---|---|
| `--test-data` | Test dataset directory | `$TEST_DATASET_PATH` |
| `--model` | Saved model file | `$MODEL_SAVE_FOLDER/model.joblib` |

### inference

Classify a single image.

| Option | Description | Default |
|---|---|---|
| `--image` | Image to classify (**required**) | none |
| `--model` | Saved model file | `$MODEL_SAVE_FOLDER/model.joblib` |

### preprocess-dataset

Preprocess every image in a dataset and save the results to disk, so later runs can skip preprocessing.

| Option | Description | Default |
|---|---|---|
| `--data` | Raw dataset to preprocess (**required**) | none |
| `--output` | Output directory for preprocessed images (**required**) | none |


### visualize-features

Visualize feature extraction overlays and histograms on an image.

| Option | Description | Default |
|---|---|---|
| `--image` | Image to visualize (**required**) | none |
| `--output-image` | Optional path to save the visualization | `$RUN_DATA_SAVE_FOLDER` |


### visualize-preprocessing

Visualize the effect of each preprocessing step on an image.

| Option | Description | Default |
|---|---|---|
| `--image` | Image to visualize (**required**) | none |
| `--output-image` | Optional path to save the visualization | `$RUN_DATA_SAVE_FOLDER` |


### generate-schema

Generate a JSON schema for the YAML configuration file. This command does not take `--config`.

| Option | Description | Default |
|---|---|---|
| `--output` | Output path for the schema file | `$CONFIG_FOLDER/run_schema.json` |
