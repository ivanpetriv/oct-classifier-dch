import numpy as np
import numpy.typing as npt
from pytest_benchmark.fixture import BenchmarkFixture

from pipeline.feature_extraction.cdch import Cdch
from pipeline.feature_extraction.hog import Hog
from pipeline.feature_extraction.lbp import Lbp
from pipeline.feature_extraction.ldch import Ldch
from pipeline.feature_extraction.ldch_py import LdchPy


def test_ldch_speed(
    benchmark: BenchmarkFixture, benchmark_images: npt.NDArray[np.uint8]
):
    extractor = Ldch(
        sections=128,
        layout="row",
        pixel_range=(1, 200),
        bins=64,
    )

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)


def test_ldch_py_speed(
    benchmark: BenchmarkFixture, benchmark_images: npt.NDArray[np.uint8]
):
    extractor = LdchPy(
        sections=128,
        layout="row",
        pixel_range=(1, 200),
        bins=64,
    )

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)


def test_cdch_speed(
    benchmark: BenchmarkFixture, benchmark_images: npt.NDArray[np.uint8]
):
    extractor = Cdch(
        5,
        (0.44, 0.55),
        (30, 222),
        256,
    )

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)


def test_lbp_speed(
    benchmark: BenchmarkFixture, benchmark_images: npt.NDArray[np.uint8]
):
    extractor = Lbp(
        16,
        3,
        "nri_uniform",
        "l1",
        46,
    )

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)


def test_hog_speed(
    benchmark: BenchmarkFixture, benchmark_images: npt.NDArray[np.uint8]
):
    extractor = Hog(
        10,
        (16, 16),
        (3, 3),
        "L2-Hys",
        False,
    )

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)


def test_loop_speed(
    benchmark: BenchmarkFixture, benchmark_images: npt.NDArray[np.uint8]
):
    """Tests the time it takes to only loop through all images."""

    class NoOp:
        def extract(self, image: npt.NDArray[np.uint8]):
            return None

    extractor = NoOp()

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)
