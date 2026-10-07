import os
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from rich.progress import track
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureExtractor(ABC, BaseEstimator, TransformerMixin):
    """
    Base class for feature extractors.
    """

    _hide_progress = os.environ.get("DISABLE_EXTRACTOR_BARS", "0") == "1"

    @abstractmethod
    def extract(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Converts an image to a histogram.
        Args:
            image (np.ndarray): 3D tensor representing image pixels (width*height*channels).
        Returns:
            np.ndarray: Numpy array representing the extracted features.
        """
        ...

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        all_features: list[np.ndarray] = []

        iterator = track(
            X,
            description=f"[cyan]Extracting {self.__class__.__name__}...",
            disable=self._hide_progress,
        )

        for image in iterator:
            image_features = self.extract(image)
            all_features.append(image_features.ravel())

        return np.array(all_features)
