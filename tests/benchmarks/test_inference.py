from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from config.oct_config import OctConfig
from pipeline._pipeline_builders.preprocessing_builder import (
    create_preprocessors_from_config,
)
from pipeline.inference.inference import Inferrer
from pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline

MODEL_LDCH = joblib.load("models/model_ldch.joblib")
MODEL_CDCH = joblib.load("models/model_cdch.joblib")
MODEL_LBP = joblib.load("models/model_lbp.joblib")
MODEL_HOG = joblib.load("models/model_hog.joblib")
CONFIG_LDCH = OctConfig.from_yaml(Path("config/svm/ldch-svm.yaml"))
CONFIG_CDCH = OctConfig.from_yaml(Path("config/svm/cdch-svm.yaml"))
CONFIG_LBP = OctConfig.from_yaml(Path("config/svm/lbp-svm.yaml"))
CONFIG_HOG = OctConfig.from_yaml(Path("config/svm/hog-svm.yaml"))


def run_benchmark(
    benchmark: BenchmarkFixture,
    images: list[npt.NDArray[np.uint8]],
    inferrer: Inferrer,
) -> None:
    def run() -> None:
        for image in images:
            inferrer.infer(image)

    benchmark(run)


@pytest.fixture(scope="module")
def inferrer_ldch():
    preprocessor = PreprocessingPipeline(
        create_preprocessors_from_config(CONFIG_LDCH.preprocessing)
    )
    return Inferrer(MODEL_LDCH, CONFIG_LDCH.dataset.classes, preprocessor)


@pytest.fixture(scope="module")
def inferrer_cdch():
    preprocessor = PreprocessingPipeline(
        create_preprocessors_from_config(CONFIG_CDCH.preprocessing)
    )
    return Inferrer(MODEL_CDCH, CONFIG_CDCH.dataset.classes, preprocessor)


@pytest.fixture(scope="module")
def inferrer_lbp():
    preprocessor = PreprocessingPipeline(
        create_preprocessors_from_config(CONFIG_LBP.preprocessing)
    )
    return Inferrer(MODEL_LBP, CONFIG_LBP.dataset.classes, preprocessor)


@pytest.fixture(scope="module")
def inferrer_hog():
    preprocessor = PreprocessingPipeline(
        create_preprocessors_from_config(CONFIG_HOG.preprocessing)
    )
    return Inferrer(MODEL_HOG, CONFIG_HOG.dataset.classes, preprocessor)

def test_inference_array_ldch(
    benchmark: BenchmarkFixture,
    benchmark_images: list[npt.NDArray[np.uint8]],
    inferrer_ldch: Inferrer,
):
    run_benchmark(benchmark, benchmark_images, inferrer_ldch)


def test_inference_array_cdch(
    benchmark: BenchmarkFixture,
    benchmark_images: list[npt.NDArray[np.uint8]],
    inferrer_cdch: Inferrer,
):
    run_benchmark(benchmark, benchmark_images, inferrer_cdch)


def test_inference_array_lbp(
    benchmark: BenchmarkFixture,
    benchmark_images: list[npt.NDArray[np.uint8]],
    inferrer_lbp: Inferrer,
):
    run_benchmark(benchmark, benchmark_images, inferrer_lbp)

def test_inference_array_hog(
    benchmark: BenchmarkFixture,
    benchmark_images: list[npt.NDArray[np.uint8]],
    inferrer_hog: Inferrer,
):
    run_benchmark(benchmark, benchmark_images, inferrer_hog)
