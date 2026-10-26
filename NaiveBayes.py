"""Starter categorical Naive Bayes interfaces for CS260 HW05."""
from __future__ import annotations

import math
from Partition import FeatureValues, FeatureDomains, Partition


class NaiveBayes:
    """Add-one smoothing, natural-log scores, and deterministic predictions.

    Required attributes: log_class[k] and log_feature[feature][value][k].
    Train only from partition. Do not print or mutate the supplied data.
    """

    log_class: list[float]
    log_feature: dict[str, dict[str, list[float]]]

    def __init__(self, partition: Partition) -> None:
        """Precompute smoothed log priors and likelihoods for all declared values.

        Require K > 0, labels in range(K), and nonempty feature domains.
        Declared classes with zero training examples are allowed.
        """
        raise NotImplementedError("TODO: implement __init__")

    def count(self, feature: str, val: str, k: int) -> int:
        """Return the training count of feature=val within class k."""
        raise NotImplementedError("TODO: implement count")

    def count_class(self, k: int) -> int:
        """Return the training count for class k."""
        raise NotImplementedError("TODO: implement count_class")

    def log_prob(self, feature: str, val: str, k: int) -> float:
        """Return the add-one-smoothed natural log likelihood."""
        raise NotImplementedError("TODO: implement log_prob")

    def log_prob_class(self, k: int) -> float:
        """Return the add-one-smoothed natural log prior."""
        raise NotImplementedError("TODO: implement log_prob_class")

    def predict_log_scores(self, x_test: FeatureValues) -> list[float]:
        """Return K unnormalized log scores in class order, without printing.

        Use only precomputed probabilities. Missing features and values outside
        the training domain raise KeyError; extra feature keys are ignored.
        Do not extend domains using test data.
        """
        raise NotImplementedError("TODO: implement predict_log_scores")

    def classify(self, x_test: FeatureValues) -> int:
        """Return the largest-score class; break ties with the smallest label."""
        raise NotImplementedError("TODO: implement classify")

    def predict(self, examples: list[FeatureValues]) -> list[int]:
        """Return labels in input order; an empty list returns an empty list."""
        raise NotImplementedError("TODO: implement predict")
