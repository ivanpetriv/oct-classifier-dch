from typing import Any

from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion

from config.oct_config import FeatureExtractionOption
from pipeline.feature_extraction.base import BaseFeatureExtractor
from pipeline.feature_extraction.cdch import Cdch
from pipeline.feature_extraction.hog import Hog
from pipeline.feature_extraction.lbp import Lbp
from pipeline.feature_extraction.ldch import Ldch
from pipeline.feature_extraction.ldch_py import LdchPy
from pipeline.feature_extraction.rdch import Rdch

_FEATURE_EXTRACTION_REGISTRY: dict[str, type[BaseFeatureExtractor]] = {
    "cdch": Cdch,
    "ldch": Ldch,
    "ldch_py": LdchPy,
    "rdch": Rdch,
    "lbp": Lbp,
    "hog": Hog,
}


def _get_feature_extraction(name: str, **kwargs: Any) -> BaseFeatureExtractor:
    extractor = _FEATURE_EXTRACTION_REGISTRY.get(name.lower())
    if not extractor:
        valid_options = ", ".join(_FEATURE_EXTRACTION_REGISTRY.keys())
        raise ValueError(
            f"Feature extractor '{name}' not found. Available: {valid_options}"
        )
    return extractor(**kwargs)


def create_feature_extraction_union_from_config(
    pipeline_config: list[FeatureExtractionOption],
) -> BaseEstimator:

    pipeline_steps: list[tuple[str, BaseEstimator]] = []

    for i, step_model in enumerate(pipeline_config):
        step_dict = step_model.model_dump(exclude_none=True)
        for step_name, params in step_dict.items():
            step_object = _get_feature_extraction(step_name, **params)
            pipeline_steps.append((f"{step_name}_{i}", step_object))

    if len(pipeline_steps) == 1:
        return pipeline_steps[0][1]

    return FeatureUnion(pipeline_steps)


def create_feature_extractors_from_config(
    pipeline_config: list[FeatureExtractionOption],
) -> list[BaseFeatureExtractor]:

    pipeline_steps: list[BaseFeatureExtractor] = []

    for step_model in pipeline_config:
        step_dict = step_model.model_dump(exclude_none=True)
        for step_name, params in step_dict.items():
            step_object = _get_feature_extraction(step_name, **params)
            pipeline_steps.append(step_object)

    return pipeline_steps
