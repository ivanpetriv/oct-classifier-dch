from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .preprocessing_step import PreprocessingStep


class PreprocessingPipeline:
    def __init__(
        self,
        steps: list[PreprocessingStep],
        hooks: dict[str, list[Callable[..., None]]] | None = None,
    ):
        self.steps = steps
        self.hooks = hooks or {}

    def add_hook(self, name: str, func: Callable[..., None]) -> None:
        self.hooks.setdefault(name, []).append(func)

    def remove_hook(self, name: str) -> None:
        self.hooks.pop(name, None)

    def _emit(self, name: str, **data: Any) -> None:
        for fn in self.hooks.get(name, []):
            fn(**data)

    def execute(self, image_array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Executes image preprocessing on a given image array.

        :param image_array: NumPy array.
        :return: A preprocessed NumPy array.
        """

        for i, step in enumerate(self.steps):
            image_array = step.preprocess(image_array)

            self._emit("after_step", step=step, index=i, image=image_array)

        return image_array
