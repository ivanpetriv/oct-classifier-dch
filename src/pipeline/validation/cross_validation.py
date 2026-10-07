import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from dataset.oct_dataset import DataLoader
from services.save_run_data import save_run_data


class CrossValidation:
    def __init__(
        self,
        pipeline: Pipeline,
        dataset: DataLoader,
        random_state: int | None,
    ):

        self.pipeline = pipeline
        self.dataset = dataset
        self.random_state = random_state

    def run(self, n_splits: int, n_folds_to_use: int | None) -> float:
        if n_splits < 2:
            raise ValueError("The number of splits must be at least 2")

        if n_folds_to_use is None:
            n_folds_to_use = n_splits
        elif not (1 <= n_folds_to_use <= n_splits):
            raise ValueError(
                f"The number of folds to use must be between 1 and {n_splits}"
            )

        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=self.random_state is not None,
            random_state=self.random_state,
        )

        all_splits = list(
            cv.split(
                self.dataset.image_paths,
                self.dataset.labels,
                groups=self.dataset.patient_ids,
            )
        )

        folds_to_run = list(range(n_splits))[:n_folds_to_use]

        all_val_targets = []
        all_val_preds = []

        for fold_idx in folds_to_run:
            self._run_single_fold(
                fold_idx=fold_idx,
                splits=all_splits[fold_idx],
                all_val_targets=all_val_targets,
                all_val_preds=all_val_preds,
            )

        # Calculate the final score to return to Optuna
        overall_f1 = f1_score(all_val_targets, all_val_preds, average="macro")

        return float(overall_f1)

    def _run_single_fold(
        self,
        fold_idx: int,
        splits: tuple[np.ndarray, np.ndarray],
        all_val_targets: list,
        all_val_preds: list,
    ) -> None:
        """Handles the lifecycle of a single fold, including creating a fresh model."""
        train_indices, val_indices = splits

        self._check_leakage(train_indices, val_indices)

        X_train, y_train = self.dataset.get_batch(
            train_indices, description=f"Fold {fold_idx + 1} Train"
        )
        X_val, y_val = self.dataset.get_batch(
            val_indices, description=f"Fold {fold_idx + 1} Val"
        )

        self.pipeline.fit(X_train, y_train)

        val_preds = self.pipeline.predict(X_val)

        all_val_targets.extend(y_val)
        all_val_preds.extend(val_preds)

    def _check_leakage(
        self, train_indices: np.ndarray, val_indices: np.ndarray
    ) -> None:
        """Ensures patients in the validation set do not appear in the training set."""
        train_patients = set(self.dataset.patient_ids[train_indices])
        val_patients = set(self.dataset.patient_ids[val_indices])
        overlap = train_patients & val_patients

        if overlap:
            raise RuntimeError(f"Patient leakage detected: {sorted(overlap)[:10]}")
