import numpy as np
import numpy.typing as npt
from skimage.restoration import denoise_wavelet, estimate_sigma  # type: ignore

from .preprocessing_step import PreprocessingStep


class Dwt(PreprocessingStep):
    def __init__(
        self,
        wavelet: str,
        mode: str,
        wavelet_levels: int,
        method: str,
        rescale_sigma: bool,
        channel_axis: int | None,
    ):
        """
        Initializes the Discrete Wavelet Transform (DWT) denoiser.

        :param wavelet: The type of wavelet to use (e.g., 'sym4', 'db2', 'haar').
        :param mode: Thresholding mode ('soft' or 'hard'). Soft is generally better for preserving edges.
        :param wavelet_levels: Number of decomposition levels. Higher levels remove more low-frequency noise but can blur.
        :param method: Method to calculate the threshold ('BayesShrink' or 'VisuShrink').
        :param rescale_sigma: Whether to rescale the image variance before denoising (improves stability).
        :param channel_axis: Set to None for grayscale, or -1 for RGB images.
        """
        self.wavelet = wavelet
        self.mode = mode
        self.wavelet_levels = wavelet_levels
        self.method = method
        self.rescale_sigma = rescale_sigma
        self.channel_axis = channel_axis

    def preprocess(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        # TODO: Check if the image is 0-255
        img_float = image.astype(np.float32) / 255.0

        sigma_est = estimate_sigma(
            img_float, channel_axis=self.channel_axis, average_sigmas=True
        )

        clean_float = denoise_wavelet(
            img_float,
            wavelet=self.wavelet,
            mode=self.mode,
            sigma=sigma_est,
            wavelet_levels=self.wavelet_levels,
            method=self.method,
            rescale_sigma=self.rescale_sigma,
            channel_axis=self.channel_axis,
        )

        return np.clip(clean_float * 255.0, 0, 255).astype(np.uint8)
