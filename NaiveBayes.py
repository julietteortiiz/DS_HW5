"""Starter categorical Naive Bayes interfaces for CS260 HW05."""
from __future__ import annotations

import math
from Partition import FeatureValues, FeatureDomains, Partition



class NaiveBayes:
    """Add-one smoothing, natural-log scores, and deterministic predictions.

    Required attributes: log_class[k] and log_feature[feature][value][k].
    Train only from n5 partition. Do not print or mutate the supplied data.
    """
    log_class: list[float]
    log_feature: dict[str, dict[str, list[float]]]

    def __init__(self, partition: Partition) -> None:
        """Precompute smoothed log priors and likelihoods for all declared values.

        Require K > 0, labels in range(K), and nonempty feature domains.
        Declared classes with zero training examples are allowed.

        1. 
        """
        self.partition = partition
        self.n = len(partition.data)
        self.log_class = []
        self.log_feature = dict()
        self.N_k = dict()

        for feature in partition.F:
            self.log_feature[feature] = dict()
            for val in partition.F.get(feature):
                self.log_feature[feature][val] = list()
 

        for k in range(partition.K):
            self.N_k[k] = self.count_class(k)
            self.log_class.append(self.log_prob_class(k))
            for feature in partition.F:
                vals = partition.F.get(feature)
                for val in vals:
                    likelihood = self.log_prob(feature, val, k)
                    self.log_feature[feature][val].append(likelihood)

    def count(self, feature: str, val: str, k: int) -> int:
        """Return the training count of feature=val within class k."""
        count = 0

        for example in self.partition.data:
            if int(example.label) == k:       
                if example.features.get(feature) == val:
                    count = count + 1
        return count
        
    def count_class(self, k: int) -> int:
        """Return the training count for class k."""
        count = 0

        for example in self.partition.data:
                    if int(example.label) == k:  
                         count = count + 1 

        return count

    def log_prob(self, feature: str, val: str, k: int) -> float:
        """Return the add-one-smoothed natural log likelihood."""
        N_k_j_v = self.count(feature, val, k)
        Nk = self.N_k.get(k)
        fj = len(self.partition.F.get(feature))
        return math.log(N_k_j_v + 1) - math.log(Nk + fj)


    def log_prob_class(self, k: int) -> float:
        """Return the add-one-smoothed natural log prior."""
        Nk = self.N_k.get(k)
        return math.log(Nk + 1) - math.log(self.n + self.partition.K)


    def predict_log_scores(self, x_test: FeatureValues) -> list[float]:
        """Return K unnormalized log scores in class order, without printing.

        Use only precomputed probabilities. Missing features and values outside
        the training domain raise KeyError; extra feature keys are ignored.
        Do not extend domains using test data.
        """
        scores = []
        for k in range(self.partition.K):
            score = self.log_class[k]
            for feature in self.partition.F:
                val = x_test[feature]
                score += self.log_feature[feature][val][k]
            scores.append(score)

        return scores



    def classify(self, x_test: FeatureValues) -> int:
        """Return the largest-score class; break ties with the smallest label."""
        scores = self.predict_log_scores(x_test)
        label = 0
        max = scores[0]
        for i in range(len(scores)):
            if scores[i] == max:
                continue
            if scores[i] > max:
                max = scores[i]
                label = i
        return label
        
    def predict(self, examples: list[FeatureValues]) -> list[int]:
        """Return labels in input order; an empty list returns an empty list."""
        predictions = list()
        for example in examples:
            predictions.append(self.classify(example))
        return predictions

