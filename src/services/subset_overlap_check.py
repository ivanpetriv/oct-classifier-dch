import os
from pathlib import Path


def check_overlap_between_subsets(subset_1_path: Path, subset_2_path: Path) -> int:
    """Checks the two subsets (train and test) for patient overlap.
    Works only if the image name has a format CLASS-PATIENT_ID-ORDER.

    Args:
        subset_1_path (Path): Path to the first subset.
        subset_2_path (Path): Path to the second subset.

    Returns:
        bool: Number of overlapped patients.
    """

    def get_patients(directory: Path):
        """Recursively gets all images from a folder."""
        patients: set[str] = set()
        for _, _, files in os.walk(directory):
            for file in files:
                if "-" in file:
                    # Extracts the ID from the filename: CLASS-PATIENT_ID-ORDER
                    patients.add(file.split("-")[1])
        return patients

    overlap = get_patients(subset_1_path) & get_patients(subset_2_path)
    return len(overlap)
