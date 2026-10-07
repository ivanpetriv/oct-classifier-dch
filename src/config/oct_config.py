from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Discriminator, Tag

from config.classification_options import (
    KNNConfig,
    LRConfig,
    RandomForestConfig,
    SVMConfig,
)
from config.dataset import Dataset
from config.dimensionality_reduction_options import PCAConfig
from config.feature_extraction_options import (
    CDCHConfig,
    HOGConfig,
    LBPConfig,
    LDCHConfig,
    RDCHConfig,
)
from config.preprocessing_options import (
    ClaheConfig,
    CornerArtifactRemoverConfig,
    CropperConfig,
    DwtDenoiserConfig,
    ResizerConfig,
    RpeRoiAlignerConfig,
)


def get_config_discriminator(value: dict[str, Any] | Any) -> str:
    """Extract discriminator key from config dict or object."""
    if isinstance(value, dict):
        return next(iter(value.keys()))
    return value.__class__.__name__.replace("Config", "").lower()


PreprocessingOption = Annotated[
    Annotated[CropperConfig, Tag("cropper")]
    | Annotated[CornerArtifactRemoverConfig, Tag("corner_artifact_remover")]
    | Annotated[DwtDenoiserConfig, Tag("dwt_denoiser")]
    | Annotated[ClaheConfig, Tag("clahe")]
    | Annotated[RpeRoiAlignerConfig, Tag("rpe_roi_aligner")]
    | Annotated[ResizerConfig, Tag("resizer")],
    Discriminator(get_config_discriminator),
]
FeatureExtractionOption = Annotated[
    Annotated[HOGConfig, Tag("hog")]
    | Annotated[LBPConfig, Tag("lbp")]
    | Annotated[CDCHConfig, Tag("cdch")]
    | Annotated[LDCHConfig, Tag("ldch")]
    | Annotated[RDCHConfig, Tag("rdch")],
    Discriminator(get_config_discriminator),
]
DimensionalityReductionOption = Annotated[
    Annotated[PCAConfig, Tag("pca")],
    Discriminator(get_config_discriminator),
]
ClassificationOption = Annotated[
    Annotated[SVMConfig, Tag("svm")]
    | Annotated[RandomForestConfig, Tag("random_forest")]
    | Annotated[KNNConfig, Tag("knn")]
    | Annotated[LRConfig, Tag("logistic_regression")],
    Discriminator(get_config_discriminator),
]


class OctConfig(BaseModel):
    """
    Model that validates a YAML file against its internal rules.
    """

    model_config = ConfigDict(extra="forbid")

    random_state: int | None
    dataset: Dataset
    preprocessing: list[PreprocessingOption] | None = None
    feature_extraction: list[FeatureExtractionOption] | None = None
    dimensionality_reduction: list[DimensionalityReductionOption] | None = None
    classification: list[ClassificationOption]

    @classmethod
    def from_yaml(cls, filepath: Path) -> "OctConfig":
        """Creates a Python object from a YAML file.
        Args:
            filepath (Path): Path to the YAML file.
        Returns:
            OctConfig: Python object that represents YAML data.
        """
        with open(filepath, "r") as file:
            config_dict = yaml.safe_load(file)

        return cls.model_validate(config_dict)

    def print_summary(self):
        print(self.model_dump_json(indent=4, exclude_unset=True))
