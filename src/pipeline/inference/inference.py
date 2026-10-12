import numpy as np
import numpy.typing as npt
from sklearn.pipeline import Pipeline

from pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline


class Inferrer:
    """Infers image class from an image with a trained model."""

    def __init__(
        self,
        model: Pipeline,
        classes: dict[str, str],
        preprocessing: PreprocessingPipeline | None,
    ):
        """Infers image class from an image with a trained model.

        Args:
            model (Pipeline): Trained model.
            classes (dict[str, str]): Classes mapping.
            preprocessing (PreprocessingPipeline | None): Preprocessing pipeline.
        """
        self.preprocessing = preprocessing
        self.model = model
        self.idx_to_class = {i: v for i, v in enumerate(classes.values())}

    def infer(self, image: npt.NDArray[np.uint8]) -> str:
        """Performs an inference on an image."""
        if self.preprocessing is not None:
            image = self.preprocessing.execute(image)

        prediction_idx = self.model.predict([image])[0]
        predicted_class = self.idx_to_class[int(prediction_idx)]

        return predicted_class
