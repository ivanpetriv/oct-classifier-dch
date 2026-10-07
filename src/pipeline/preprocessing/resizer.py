from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from .preprocessing_step import PreprocessingStep


class Resizer(PreprocessingStep):
    def __init__(self, target_size: tuple[int, int]):
        self.target_size = target_size

    def preprocess(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return cast(
            npt.NDArray[np.uint8],
            cv2.resize(
                image,
                self.target_size,
                interpolation=cv2.INTER_LANCZOS4,
            ),
        )
