import numpy as np
import numpy.typing as npt
import oct_rust

from .base import BaseFeatureExtractor


class Cdch(BaseFeatureExtractor):
    """
    Circular Distributed Cumulative Histogram (CDCH) extractor.
    It divides the image radially, then creates a vector of intensities for each ray.
    """

    def __init__(
        self,
        angle_step: int,
        center: tuple[float, float] | None,
        pixel_range: tuple[int, int],
        bins: int,
    ):
        """
        Initializes the CDCH.

        Args:
            angle_step: The step size in degrees for casting rays (0-360).
            center: The coordinates of the center point [0; 1], or None for center of mass.
            pixel_range: Tuple specifying valid intensities (min, max).
            bins: Number of bins for the histogram.
        """
        # TODO: fill in valueerrors
        if not (0 < angle_step <= 360):
            raise ValueError("")
        if center and not ((0 <= center[0] <= 1) and (0 <= center[1] <= 1)):
            raise ValueError("")
        if not (0 <= pixel_range[0] <= pixel_range[1] <= 255):
            raise ValueError("")
        if not (0 < bins):
            raise ValueError("")

        self.angle_step = angle_step
        self.center = center
        self.pixel_range = pixel_range
        self.bins = bins

    def extract(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        clean_image = np.ascontiguousarray(image, dtype=np.uint8)

        return oct_rust.extract_cdch(
            clean_image,
            self.angle_step,
            self.center[0] if self.center is not None else None,
            self.center[1] if self.center is not None else None,
            self.pixel_range[0],
            self.pixel_range[1],
            self.bins,
        )
