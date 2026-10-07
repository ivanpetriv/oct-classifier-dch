from typing import Annotated, Literal

from pydantic import BaseModel, Field

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class SVMLinearParams(BaseModel):
    kernel: Literal["linear"]
    C: PositiveFloat
    shrinking: bool
    class_weight: str | dict[int, int] | None
    n_jobs: int | None = None


class SVMRbfParams(BaseModel):
    kernel: Literal["rbf"]
    C: PositiveFloat
    gamma: str | NonNegativeFloat
    shrinking: bool
    class_weight: str | dict[int, int] | None
    n_jobs: int | None = None


class SVMPolyParams(BaseModel):
    kernel: Literal["poly"]
    C: PositiveFloat
    gamma: str | NonNegativeFloat
    degree: NonNegativeInt
    shrinking: bool
    class_weight: str | dict[int, int] | None
    n_jobs: int | None = None


class SVMConfig(BaseModel):
    svm: Annotated[
        SVMLinearParams | SVMRbfParams | SVMPolyParams, Field(discriminator="kernel")
    ]


class RandomForestParams(BaseModel):
    n_estimators: PositiveInt
    max_depth: PositiveInt | None = None
    min_samples_split: Annotated[int, Field(ge=2)]
    min_samples_leaf: PositiveInt
    max_features: Literal["sqrt", "log2"] | None
    bootstrap: bool
    max_samples: Annotated[float, Field(gt=0.0, le=1.0)] | None
    criterion: Literal["gini", "entropy", "log_loss"]
    class_weight: Literal["balanced", "balanced_subsample"] | None
    random_state: int | None
    n_jobs: int | None = None


class RandomForestConfig(BaseModel):
    random_forest: RandomForestParams


class KNNParams(BaseModel):
    n_neighbors: PositiveInt
    weights: Literal["uniform", "distance"]
    p: Annotated[int, Field(ge=1)]
    metric: Literal["minkowski", "cosine"]
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: PositiveInt


class KNNConfig(BaseModel):
    knn: KNNParams


class LRParams(BaseModel):
    C: PositiveFloat
    class_weight: Literal["balanced"] | None
    solver: Literal["lbfgs", "saga"]
    max_iter: PositiveInt
    random_state: int | None


class LRConfig(BaseModel):
    logistic_regression: LRParams
