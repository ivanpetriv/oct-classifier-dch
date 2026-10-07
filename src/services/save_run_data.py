import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import classification_report  # type: ignore

from config.oct_config import OctConfig


def save_run_data(
    output_dir: Path,
    config: OctConfig,
    targets: NDArray[np.uint8] | list[int] | None,
    preds: NDArray[np.uint8] | list[int] | None,
) -> None:
    """Saves the configuration and results into a single, timestamped JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"run_{timestamp}.json"

    config_dict = json.loads(config.model_dump_json())

    payload: dict[str, Any] = {"timestamp": timestamp, "configuration": config_dict}

    if targets is not None and preds is not None:
        # Convert to numpy arrays
        targets = np.asarray(targets, dtype=np.uint8).ravel()  # Ensure 1D
        preds = np.asarray(preds, dtype=np.uint8).ravel()  # Ensure 1D

        # Validate shapes
        if targets.shape != preds.shape:
            raise ValueError(
                f"Shape mismatch: targets {targets.shape} vs preds {preds.shape}"
            )

        # Validate no NaN values
        if np.isnan(targets).any() or np.isnan(preds).any():
            raise ValueError("targets or preds contain NaN values")

        # Get valid class indices
        num_classes = len(config.dataset.classes)

        if targets.max() >= num_classes or preds.max() >= num_classes:
            raise ValueError(
                f"Class index out of range. Max: {max(targets.max(), preds.max())}, "
                f"but only {num_classes} classes defined"
            )

        # Build target_names
        if isinstance(config.dataset.classes, list):
            target_names = config.dataset.classes
        else:
            idx_to_class = {v: k for k, v in config.dataset.classes.items()}
            target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

        # Generate report
        report_dict = classification_report(
            targets,
            preds,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        payload["results"] = report_dict

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=4, default=str)

    print(f"Run data successfully saved to: {filepath}")
