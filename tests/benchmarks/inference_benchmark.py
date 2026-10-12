from pathlib import Path

import joblib  # type: ignore
import numpy as np
import numpy.typing as npt
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from pipeline._pipeline_builders.preprocessing_builder import (
    create_preprocessors_from_config,
)
from pipeline.inference.inference import Inferrer
from pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline

classes = {"DME": "DME", "CNV": "CNV", "NORMAL": "NORMAL", "DRUSEN": "DRUSEN"}
MODEL_PATH_LDCH = Path("models/model_ldch.joblib")
MODEL_PATH_CDCH = Path("models/model_cdch.joblib")
MODEL_PATH_LBP = Path("models/model_lbp.joblib")
MODEL_PATH_HOG = Path("models/model_hog.joblib")
MODEL_LDCH = joblib.load(MODEL_PATH_LDCH)


@pytest.fixture(scope="module")
def preprocessor() -> PreprocessingPipeline:
    return PreprocessingPipeline(create_preprocessors_from_config(config.preprocessing))


@pytest.fixture(scope="module")
def inferrer_ldch(preprocessor: PreprocessingPipeline):
    return Inferrer(MODEL_LDCH, classes, preprocessor)


def test_inference_array_ldch(
    benchmark: BenchmarkFixture,
    benchmark_images: list[npt.NDArray[np.uint8]],
    inferrer_ldch: Inferrer,
):
    def run():
        for image in benchmark_images:
            # convert to numpy
            inferrer_ldch.infer(image)

    benchmark(run)
