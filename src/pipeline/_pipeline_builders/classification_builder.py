import inspect
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from config.oct_config import ClassificationOption

# All classifiers
_CLASSIFICATION_REGISTRY: dict[str, type[BaseEstimator]] = {
    "svm": SVC,
    "svm_linear": LinearSVC,
    "random_forest": RandomForestClassifier,
    "knn": KNeighborsClassifier,
    "logistic_regression": LogisticRegression,
}


def _get_classification(name: str, **kwargs: Any) -> BaseEstimator:
    """Instantiates the classifier with the given params."""
    model_class = _CLASSIFICATION_REGISTRY.get(name.lower())
    if not model_class:
        valid_options = ", ".join(_CLASSIFICATION_REGISTRY.keys())
        raise ValueError(f"Classifier '{name}' not found. Available: {valid_options}")
    return model_class(**kwargs)


def create_classification_step(
    classification_config: list[ClassificationOption],
    random_state: int | None,
) -> BaseEstimator:
    """
    Retrieves the params from the config and instantiates the classifier object.
    Args:
        classification_config (list[ClassificationOption]): Configuration.
        random_state (int | None): Random seed to supply the classifier.
    Returns:
        BaseEstimator: Classifier object.
    """
    if not classification_config:
        raise ValueError("Classification config cannot be empty")

    classifier_model = classification_config[0]
    classifier_dict = classifier_model.model_dump(exclude_none=True)

    for name, params in classifier_dict.items():
        model_class = _CLASSIFICATION_REGISTRY.get(name.lower())

        # Safely inject the global random state only if the specific
        # sklearn classifier accepts it in its constructor and it wasn't defined in YAML
        if (
            model_class
            and "random_state" in inspect.signature(model_class).parameters
            and "random_state" not in params
        ):
            params["random_state"] = random_state

        return _get_classification(name, **params)
