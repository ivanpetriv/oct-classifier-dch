from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NormalFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class HOGParams(BaseModel):
    orientations: Annotated[int, Field(ge=1, le=36)]
    pixels_per_cell: tuple[PositiveInt, PositiveInt]
    cells_per_block: tuple[PositiveInt, PositiveInt]
    transform_sqrt: bool
    block_norm: Literal["L1", "L1-sqrt", "L2", "L2-Hys"]


class HOGConfig(BaseModel):
    hog: HOGParams


class LBPParams(BaseModel):
    n_points: PositiveInt
    radius: PositiveInt
    method: Literal["default", "ror", "uniform", "nri_uniform", "var"]
    norm: Literal["l1", "l2"]
    padding_threshold: NonNegativeInt


class LBPConfig(BaseModel):
    lbp: LBPParams


class CDCHParams(BaseModel):
    center: tuple[NormalFloat, NormalFloat]
    pixel_range: tuple[NonNegativeInt, NonNegativeInt]
    angle_step: Annotated[int, Field(ge=1, le=360)]
    bins: PositiveInt

    @model_validator(mode="after")
    def validate_pixel_range(self):
        if self.pixel_range[0] > self.pixel_range[1]:
            raise ValueError(
                "pixel_range minimum must be less than or equal to maximum"
            )
        return self


class CDCHConfig(BaseModel):
    cdch: CDCHParams


class LDCHParams(BaseModel):
    sections: PositiveInt
    layout: Literal["row", "col"]
    pixel_range: tuple[NonNegativeInt, NonNegativeInt]
    bins: PositiveInt

    @model_validator(mode="after")
    def validate_pixel_range(self):
        if self.pixel_range[0] > self.pixel_range[1]:
            raise ValueError(
                "pixel_range minimum must be less than or equal to maximum"
            )
        return self


class LDCHConfig(BaseModel):
    ldch: LDCHParams


class RDCHParams(BaseModel):
    ring_step: PositiveInt
    center: tuple[NonNegativeFloat, NonNegativeFloat]
    pixel_range: tuple[NonNegativeInt, NonNegativeInt]
    bins: PositiveInt

    @model_validator(mode="after")
    def validate_pixel_range(self):
        if self.pixel_range[0] > self.pixel_range[1]:
            raise ValueError(
                "pixel_range minimum must be less than or equal to maximum"
            )
        return self


class RDCHConfig(BaseModel):
    rdch: RDCHParams
