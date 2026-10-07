import re
from pathlib import Path

import numpy as np
from PIL import Image
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline


class DataLoader:
    allowed_image_extensions: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

    def __init__(
        self,
        dataset_path: Path,
        allowed_classes: list[str] | dict[str, str],
        samples_per_class: int | None,
        random_state: int | None,
        preprocessor: PreprocessingPipeline | None,
    ):
        """Initializes the dataset.

        Args:
            dataset_path (Path): Path to the dataset. Must contain disease subfolders.
            allowed_classes (list[str] | dict[str, str]): Classes to load.
                If a list, maps 1:1.
                If a dict, maps physical directory names to logical class names.
            samples_per_class (int | None, optional): How many items per logical class.
        """
        self.path = dataset_path
        self.samples_per_class = samples_per_class

        if isinstance(allowed_classes, list):
            self.dir_to_logical = {c: c for c in allowed_classes}
        else:
            self.dir_to_logical = allowed_classes

        self.logical_classes = sorted(self.dir_to_logical.values())

        self.class_mapping = {c: i for i, c in enumerate(self.logical_classes)}

        # Index the dataset
        (
            self.image_paths,
            self.labels,
            self.class_names,
            self.patient_ids,
        ) = self._index_dataset()

        self.preprocessor = preprocessor

    def _extract_patient_id(self, path: Path) -> str:
        """Extracts patient ID from filename.

        Assumes format: `<disease>-<patientid>-<number>` for Ravelli dataset.
        Falls back to filename stem if format doesn't match.
        """
        match = re.match(r"^[^-]+-(\d+)-\d+$", path.stem)
        return match.group(1) if match else path.stem

    def _index_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Index dataset with round-robin sampling per logical class."""
        image_paths: list[Path] = []
        labels: list[int] = []
        class_names: list[str] = []
        patient_ids: list[str] = []

        # Organize: logical_class -> patient_id -> [image_paths]
        logical_to_patients: dict[str, dict[str, list[Path]]] = {
            c: {} for c in self.logical_classes
        }

        # Scan directories and group by logical class and patient
        for dir_name, logical_class in self.dir_to_logical.items():
            class_dir = self.path / dir_name
            if not class_dir.exists():
                print(f"Warning: Disease directory '{class_dir}' does not exist.")
                continue

            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in self.allowed_image_extensions:
                    continue

                patient_id = self._extract_patient_id(image_path)
                logical_to_patients[logical_class].setdefault(patient_id, []).append(
                    image_path
                )

        # Round-robin extraction per logical class
        for logical_class in self.logical_classes:
            label_idx = self.class_mapping[logical_class]
            patient_to_images = logical_to_patients[logical_class]
            active_patients = list(patient_to_images.keys())

            valid_samples = 0
            while active_patients:
                if self.samples_per_class and valid_samples >= self.samples_per_class:
                    break

                for patient_id in list(active_patients):
                    if (
                        self.samples_per_class
                        and valid_samples >= self.samples_per_class
                    ):
                        break

                    img_path = patient_to_images[patient_id].pop(0)
                    image_paths.append(img_path)
                    labels.append(label_idx)
                    class_names.append(logical_class)
                    patient_ids.append(patient_id)
                    valid_samples += 1

                    # Remove patient if out of images
                    if not patient_to_images[patient_id]:
                        active_patients.remove(patient_id)

        return (
            np.array(image_paths),
            np.array(labels, dtype=np.uint8),
            np.array(class_names),
            np.array(patient_ids),
        )

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a single image."""
        with Image.open(image_path) as img:
            img = img.convert("L")
            arr = np.array(img)

            if self.preprocessor is not None:
                arr = self.preprocessor.execute(arr)

        return np.ascontiguousarray(arr, dtype=np.uint8)

    def get_batch(
        self, indices: np.ndarray, description: str = "Loading Data"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load and preprocess images for given indices."""
        paths = self.image_paths[indices]
        labels = self.labels[indices]
        classes = self.class_names[indices]

        X_data: list[np.ndarray] = []
        y_data: list[int] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[yellow]{task.fields[current_class]}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"[green]{description}", total=len(paths), current_class=""
            )
            current_displayed_class = ""

            for image_path, label_idx, class_name in zip(paths, labels, classes):
                if class_name != current_displayed_class:
                    progress.update(task, current_class=f"({class_name})")
                    current_displayed_class = class_name

                image_array = self._load_image(image_path)
                X_data.append(image_array)
                y_data.append(label_idx)

                progress.advance(task)

        if not X_data:
            raise ValueError("No data could be gathered for the given indices.")

        return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.uint8)
