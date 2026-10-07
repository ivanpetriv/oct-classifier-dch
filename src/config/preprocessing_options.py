from typing import Annotated

from pydantic import BaseModel, Field

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonEmptyStr = Annotated[str, Field(min_length=1)]


class CropperParams(BaseModel):
    top: NonNegativeInt
    bottom: NonNegativeInt
    left: NonNegativeInt
    right: NonNegativeInt


class CropperConfig(BaseModel):
    cropper: CropperParams


class CornerArtifactRemoverParams(BaseModel):
    min_threshold: Annotated[int, Field(ge=0, le=255)]
    approximation_epsilon: PositiveFloat
    max_area_ratio: Annotated[float, Field(ge=0.0, le=1.0)]
    ksize: tuple[PositiveInt, PositiveInt]


class CornerArtifactRemoverConfig(BaseModel):
    corner_artifact_remover: CornerArtifactRemoverParams


class DwtDenoiserParams(BaseModel):
    wavelet: NonEmptyStr
    mode: NonEmptyStr
    wavelet_levels: PositiveInt
    method: NonEmptyStr
    rescale_sigma: bool
    channel_axis: int | None = Field(ge=-1)


class DwtDenoiserConfig(BaseModel):
    dwt_denoiser: DwtDenoiserParams


class ClaheParams(BaseModel):
    clip_limit: PositiveFloat
    tile_grid_size: tuple[PositiveInt, PositiveInt]


class ClaheConfig(BaseModel):
    clahe: ClaheParams


class RpeRoiAlignerParams(BaseModel):
    roi_above: NonNegativeInt
    roi_below: NonNegativeInt
    blur_sigma: PositiveInt
    max_jump: PositiveInt
    poly_degree: PositiveInt


class RpeRoiAlignerConfig(BaseModel):
    rpe_roi_aligner: RpeRoiAlignerParams


class ResizerParams(BaseModel):
    target_size: tuple[PositiveInt, PositiveInt]


class ResizerConfig(BaseModel):
    resizer: ResizerParams
