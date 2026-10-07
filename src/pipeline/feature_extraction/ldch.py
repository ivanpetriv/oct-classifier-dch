from typing import Literal

import numpy as np
import numpy.typing as npt
import oct_rust

from .base import BaseFeatureExtractor


class Ldch(BaseFeatureExtractor):
    """
    Linear Distributed Cumulative Histogram (LDCH) extractor.
    It divides the image into horizontal or vertical sections,
    then creates a vector of intensities for each section.
    """

    def __init__(
        self,
        sections: int,
        layout: Literal["row", "col"],
        pixel_range: tuple[int, int],
        bins: int,
    ):
        """
        Initializes the LDCH converter.

        :param sections: Number of sections to divide the image into.
        :param layout: Layout of the sections, either "row" for horizontal or "col" for vertical.
        :param pixel_range: Tuple specifying the range of pixel values to consider (min, max).
        :param bins: Number of bins to use for the histogram.
        """
        if sections <= 0 or bins <= 0:
            raise ValueError
        if not (0 <= pixel_range[0] <= pixel_range[1] <= 255):
            raise ValueError

        self.sections = sections
        self.layout = layout
        self.pixel_range = pixel_range
        self.bins = bins

    def extract(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        clean_image = np.ascontiguousarray(image, dtype=np.uint8)
        return oct_rust.extract_ldch(
            clean_image,
            self.sections,
            self.layout == "row",
            self.pixel_range[0],
            self.pixel_range[1],
            self.bins,
        )
