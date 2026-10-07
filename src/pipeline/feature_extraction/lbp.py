import numpy as np
import numpy.typing as npt
from skimage.feature import local_binary_pattern  # type: ignore

from .base import BaseFeatureExtractor


class Lbp(BaseFeatureExtractor):
    """
    Local Binary Patterns (LBP) feature extractor.
    Captures micro-textures of the tissue (edges, spots, flat areas).
    """

    def __init__(
        self, n_points: int, radius: int, method: str, norm: str, padding_threshold: int
    ):
        """
        Initializes the LBP extractor.
        :param n_points: Number of circularly symmetric neighbor set points.
        :param radius: Radius of circle.
        :param method: Method to determine the pattern ('uniform' reduces dimensionality).
        :param padding_threshold: Pixel intensity threshold to ignore black padding.
        """
        self.n_points = n_points
        self.radius = radius
        self.method = method
        self.norm = norm
        self.padding_threshold = padding_threshold

        if self.method == "uniform":
            self.num_bins = self.n_points + 2
        elif self.method == "nri_uniform":
            self.num_bins = self.n_points * (self.n_points - 1) + 3
        else:
            self.num_bins = 2**self.n_points

            if self.num_bins > 10000:
                raise ValueError(
                    f"Configuration n_points={self.n_points} with method='{self.method}' "
                    f"creates {self.num_bins} bins, which will exceed memory limits. "
                    "Use 'uniform' or 'nri_uniform' for high n_points."
                )

    def extract(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Extracts the LBP feature vector from the image.
        """
        # if np.issubdtype(image.dtype, np.floating):
        #     if image.max() <= 1.0:
        #         image = (image * 255.0).astype(np.uint8)
        #     else:
        #         image = image.astype(np.uint8)

        lbp_image = local_binary_pattern(image, self.n_points, self.radius, self.method)

        tissue_mask = image > self.padding_threshold
        valid_lbp_values = lbp_image[tissue_mask]

        if len(valid_lbp_values) > 0:
            hist, _ = np.histogram(
                valid_lbp_values,
                bins=self.num_bins,
                range=(0, self.num_bins),
                density=False,
            )
            hist = hist.astype(np.float64)

            hist_sum = np.sum(hist)
            if hist_sum > 0:
                if self.norm == "l1":
                    hist /= hist_sum
                elif self.norm == "l2":
                    hist /= np.sqrt(np.sum(hist**2))
                else:
                    raise ValueError(f"Unknown normalization method: {self.norm}")
        else:
            hist = np.zeros(self.num_bins, dtype=np.float64)

        return hist
