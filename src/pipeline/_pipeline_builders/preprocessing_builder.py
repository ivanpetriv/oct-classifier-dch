from typing import Any

from config.oct_config import PreprocessingOption
from pipeline.preprocessing.bilateral_denoiser import BilateralDenoiser
from pipeline.preprocessing.clahe import Clahe
from pipeline.preprocessing.cropper import Cropper
from pipeline.preprocessing.dwt import Dwt
from pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from pipeline.preprocessing.preprocessing_step import PreprocessingStep
from pipeline.preprocessing.resizer import Resizer
from pipeline.preprocessing.rpe_roi_extractor import RpeRoiExtractor
from pipeline.preprocessing.white_region_mask import WhiteRegionMask

_PREPROCESSING_REGISTRY: dict[str, type[PreprocessingStep]] = {
    "bilateral_denoiser": BilateralDenoiser,
    "dwt_denoiser": Dwt,
    "clahe_filter": Clahe,
    "corner_artifact_remover": WhiteRegionMask,
    "rpe_roi_aligner": RpeRoiExtractor,
    "resizer": Resizer,
    "cropper": Cropper,
}


def _get_preprocessing(name: str, **kwargs: Any) -> PreprocessingStep:
    preprocessor = _PREPROCESSING_REGISTRY.get(name.lower())
    if not preprocessor:
        valid_options = ", ".join(_PREPROCESSING_REGISTRY.keys())
        raise ValueError(f"Preprocessor '{name}' not found. Available: {valid_options}")
    return preprocessor(**kwargs)


def create_preprocessors_from_config(
    pipeline_config: list[PreprocessingOption] | None,
) -> list[PreprocessingStep]:
    if pipeline_config is None:
        return []

    pipeline_steps: list[PreprocessingStep] = []

    for step_model in pipeline_config:
        step_dict = step_model.model_dump()
        for step_name, params in step_dict.items():
            step_object = _get_preprocessing(step_name, **params)
            pipeline_steps.append(step_object)

    return pipeline_steps
