"""Shared data structures for CS260 HW05 (provided; do not change interfaces)."""
from __future__ import annotations

FeatureValues = dict[str, str]
FeatureDomains = dict[str, list[str]]
ConfusionMatrix = list[list[int]]


class Example:
    """One categorical example with an integer label in range(K)."""

    def __init__(self, features: FeatureValues, label: int) -> None:
        self.features: FeatureValues = features
        self.label: int = label


class Partition:
    """Examples, feature domains, and the declared number of classes."""

    def __init__(self, data: list[Example], F: FeatureDomains, K: int) -> None:
        self.data: list[Example] = data
        self.n: int = len(data)
        self.F: FeatureDomains = F
        self.K: int = K
