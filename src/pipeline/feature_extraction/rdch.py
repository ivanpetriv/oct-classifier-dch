import numpy as np
import oct_rust

from .base import BaseFeatureExtractor


class Rdch(BaseFeatureExtractor):
    """
    Ring Distributed Cumulative Histogram (RDCH) extractor.
    It divides the image into concentric rings radiating from a center point
    using a fixed step width, then creates a vector of intensities for each ring.
    """

    def __init__(
        self,
        ring_step: int,
        center: tuple[float, float] | None,
        pixel_range: tuple[int, int],
        bins: int,
    ):
        """
        Initializes the RDCH converter.

        Args:
            ring_step: The width of each concentric ring in pixels.
            center: The coordinates of the center point [0; 1]. If None, calculates center of mass.
            pixel_range: Tuple specifying valid intensities (min, max).
            bins: Number of bins to use for the histogram.
        """
        if ring_step <= 0:
            raise ValueError("")
        if center and not ((0 <= center[0] <= 1) and (0 <= center[1] <= 1)):
            raise ValueError("")
        if not (0 <= pixel_range[0] <= pixel_range[1] <= 255):
            raise ValueError("")
        if bins <= 0:
            raise ValueError("")

        self.ring_step = ring_step
        self.center = center
        self.pixel_range = pixel_range
        self.bins = bins

    def extract(self, image: np.ndarray) -> np.ndarray:
        clean_image = np.ascontiguousarray(image, dtype=np.uint8)

        return oct_rust.extract_rdch(
            clean_image,
            self.ring_step,
            self.center[0] if self.center is not None else None,
            self.center[1] if self.center is not None else None,
            self.pixel_range[0],
            self.pixel_range[1],
            self.bins,
        )
