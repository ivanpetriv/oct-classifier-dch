from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.oct_config import OctConfig
from pipeline._pipeline_builders.classification_builder import (
    create_classification_step,
)
from pipeline._pipeline_builders.dimensionality_reduction_builder import (
    create_reduction_step,
)
from pipeline._pipeline_builders.feature_extraction_builder import (
    create_feature_extraction_union_from_config,
)


def build_pipeline(config: OctConfig) -> Pipeline:
    """
    Creates a pipeline combining feature extraction, dimensionality reduction
    and classification steps.

    Args:
        config (OctConfig): Config for the pipeline.

    Returns:
        Pipeline: Object representing a scikit-learn pipeline.
    """
    pipeline: list[tuple[str, BaseEstimator]] = []

    if config.feature_extraction:
        pipeline.append(
            (
                "feature_extractor",
                create_feature_extraction_union_from_config(config.feature_extraction),
            )
        )

    pipeline.append(("scaler", StandardScaler()))

    if config.dimensionality_reduction:
        pipeline.append(
            (
                "dimensionality_reduction",
                create_reduction_step(config.dimensionality_reduction),
            )
        )

    if config.classification:
        pipeline.append(
            (
                "classifier",
                create_classification_step(
                    config.classification, random_state=config.random_state
                ),
            )
        )

    return Pipeline(pipeline, memory=None)
