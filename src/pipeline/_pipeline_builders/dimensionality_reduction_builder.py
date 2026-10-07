from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

from config.oct_config import DimensionalityReductionOption


class SafePCA(PCA):
    """Prevents crashes when n_components > min(n_samples, n_features)."""

    def _cap_components(self, X: Any) -> None:
        n_samples, n_features = X.shape
        limit = min(n_samples, n_features) - 1

        if self.n_components > limit:
            self.n_components = max(1, limit)

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> Self:
        self._cap_components(X)
        return super().fit(X, y, **fit_params)

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        self._cap_components(X)
        return super().fit_transform(X, y, **fit_params)


_REDUCTION_REGISTRY = {
    "pca": SafePCA,
}


def _get_reduction(name: str, **kwargs: Any):
    reducer_class = _REDUCTION_REGISTRY.get(name.lower())
    if not reducer_class:
        valid = ", ".join(_REDUCTION_REGISTRY.keys())
        raise ValueError(f"Reduction '{name}' not found. Available: {valid}")
    return reducer_class(**kwargs)


def create_reduction_step(
    reduction_config: list[DimensionalityReductionOption],
) -> BaseEstimator:
    reducer_model = reduction_config[0]
    reducer_dict = reducer_model.model_dump(exclude_none=True)

    # Currently supports only 1 reductor
    name, params = next(iter(reducer_dict.items()))

    if name == "pca":
        return SafePCA(**params)
    else:
        raise ValueError(f"Unknown dimensionality reduction technique: {name}")
