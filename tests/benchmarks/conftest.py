import os
from pathlib import Path

import cv2
import dotenv
import numpy as np
import numpy.typing as npt
import pytest

dotenv.load_dotenv()
cv2.setNumThreads(1)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Adds the options to the CLI."""
    parser.addoption(
        "--image-dir",
        default=os.getenv("BENCHMARK_DATASET_PATH"),
        help="Directory with benchmark images (default: BENCHMARK_DATASET_PATH).",
    )


@pytest.fixture(scope="session")
def benchmark_images(request: pytest.FixtureRequest) -> list[npt.NDArray[np.uint8]]:
    """Preloads the images specified in the CLI `--image-dir`."""

    value = request.config.getoption("--image-dir")
    if not value:
        pytest.skip("Pass --image-dir or set BENCHMARK_DATASET_PATH")
    image_dir = Path(value)
    assert image_dir.is_dir(), f"Not a directory: {image_dir}"

    # Finds all images
    paths = sorted(
        p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    assert paths, f"No images found in {image_dir}!"

    # Loads all images
    images: list[npt.NDArray[np.uint8]] = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert img is not None, f"Could not read {path}"
        images.append(np.asarray(img, dtype=np.uint8))

    return images
