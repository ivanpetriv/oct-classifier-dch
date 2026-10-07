from typing import Annotated, Literal

from pydantic import BaseModel, Field

NonNegativeInt = Annotated[int, Field(ge=0)]


class PCAParams(BaseModel):
    n_components: NonNegativeInt
    whiten: bool
    svd_solver: Literal["auto", "full", "covariance_eigh", "arpack", "randomized"]
    iterated_power: NonNegativeInt | Literal["auto"]
    power_iteration_normalizer: Literal["auto", "QR", "LU", "none"]


class PCAConfig(BaseModel):
    pca: PCAParams
