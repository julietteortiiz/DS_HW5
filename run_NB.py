"""Read categorical datasets and run CS260 HW05 from the command line.

Computational helpers return values and never print. Only main() and
print_confusion_matrix() produce output. No third-party packages are required.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Optional
import statistics

from NaiveBayes import NaiveBayes
from Partition import ConfusionMatrix, Example, FeatureDomains, FeatureValues, Partition

from typing import List
from collections import OrderedDict



def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse required train/test paths; None reads sys.argv[1:]."""
    parser = argparse.ArgumentParser(description="Run categorical Naive Bayes")
    parser.add_argument("-r", "--train_filename", required=True)
    parser.add_argument("-e", "--test_filename", required=True)
    return parser.parse_args(argv)


def read_csv(filename: str) -> Partition:
    """Read census CSV with a header; sex is the label (Male=0, Female=1).

    Exclude sex from features/F. Keep feature and first-seen value order,
    strip surrounding whitespace, and keep all feature values as strings.
    Return K=2 even if one class is absent. Invalid sex values raise ValueError.
    """
    data: List[Example] = []
    F: FeatureValues = OrderedDict()
    K = 2
    header_names: List[str] = []
    sex_index: Optional[int] = None

    with open(filename, "r") as f:
        d = csv.reader(f)

        for row_num, row in enumerate(d):

            #add header feature decs to dict
            if row_num == 0:
                header_names = [name.strip() for name in row]
                for idx, name in enumerate(header_names):
                    if name == "sex":
                        sex_index = idx
                    else:
                        F[name] = list()
                continue

            X_dict = {}
            label = -1
            for idx, name in enumerate(header_names):
                val = row[idx].strip()
                if idx == sex_index:
                    if val == "Male":
                        label = 0
                    elif val == "Female":
                        label = 1
                    else:
                        raise ValueError(f"Invalid sex value: {val}")
                    continue
                X_dict[name] = val
                if val not in F[name]:
                    F[name].append(val)
            data.append(Example(X_dict, label))

    return Partition(data, F, K)


def read_arff(filename: str) -> Partition:
    """Read dense categorical ARFF into a Partition.

    Skip blank lines and full-line % comments. Attribute/data keywords are
    case insensitive. Attribute names may be single/double quoted or bare.
    The final attribute is the class; map its declared values to 0..K-1 in
    declaration order, including classes absent from the rows. Preserve
    declared feature values as strings (including values absent from rows).
    Training/test files must declare the same class order. Sparse ARFF and
    continuous attributes are outside the scope of this homework.
    """
    arff_file = open(filename, 'r', encoding='utf-8')
    data: List[Example] = []
    F: FeatureValues = OrderedDict()

    header = arff_file.readline()
    line = arff_file.readline().strip()
    K = 0

    while line != "@data":
        line = line.replace('{','').replace('}','').replace(',','')
        tokens = line.split()
        name = tokens[1][1:-1]
        features = tokens[2:]

        # label
        if name != "class":
            F[name] = features
        else:
            K = len(features)
            first = tokens[2]
            
        line = arff_file.readline().strip()

        # read the examples
    for line in arff_file:
        tokens = line.strip().split(",")
        
        X_dict = {}
        i = 0
        for key in F:
            val = tokens[i]
            X_dict[key] = val
            i += 1
        label = tokens[-1]
        data.append(Example(X_dict, label))

    arff_file.close()

    partition = Partition(data, F, K)
    return partition


def read_data(filename: str) -> Partition:
    """Dispatch .csv/.arff files; unsupported suffixes raise ValueError."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return read_csv(filename)
    if suffix == ".arff":
        return read_arff(filename)
    raise ValueError("Expected a .csv or .arff file")


def confusion_matrix(y_true: list[int], y_pred: list[int], K: int) -> ConfusionMatrix:
    """Return integer counts with actual rows and predicted columns; never print.

    Return K x K zeros for empty inputs. Raise ValueError for unequal lengths,
    K <= 0, or labels outside range(K). Do not modify either input list.
    """
    cm = [[0 for _ in range(K)] for _ in range(K)]
    #    Predicted   C1 C2 ...
    # Actual       C1
    #              C2
    #
            
    for i in range(len(y_true)):
        k_true = int(y_true[i])
        k_pred = int(y_pred[i])
        cm[k_true][k_pred] += 1

    return cm 


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Return fraction correct; empty or unequal-length inputs raise ValueError."""
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return (correct / len(y_true))

def balanced_error_rate(matrix: ConfusionMatrix) -> float:
    """Return the mean class error rate for a square confusion matrix.

    Raise ValueError for an empty/nonsquare matrix, negative counts, or any
    actual class with zero examples (its error rate would be undefined).
    """
    error_rates = list()
    
    for i in range(len(matrix)):
        correct = matrix[i][i]
        errors = sum(matrix[i]) - correct
        error_rates.append(errors/sum(matrix[i]))

    return statistics.mean(error_rates)   
        

def print_confusion_matrix(matrix: ConfusionMatrix) -> None:
    """Provided presentation helper; computation belongs in confusion_matrix()."""
    print("Actual rows; predicted columns")
    print("     " + "".join(f"{k:6d}" for k in range(len(matrix))))
    for k, row in enumerate(matrix):
        print(f"{k:3d} |" + "".join(f"{value:6d}" for value in row))


def main(argv: Optional[list[str]] = None) -> None:
    """Provided CLI wiring; argv permits tests without changing sys.argv."""
    args = parse_args(argv)
    train = read_data(args.train_filename)
    test = read_data(args.test_filename)
    print(f"num train = {train.n}, num classes = {train.K}")
    print(f"num test  = {test.n}, num classes = {test.K}")
    model = NaiveBayes(train)
    y_true = [example.label for example in test.data]
    y_pred = model.predict([example.features for example in test.data])
    matrix = confusion_matrix(y_true, y_pred, train.K)
    correct = sum(matrix[k][k] for k in range(train.K))
    print(f"Accuracy: {accuracy(y_true, y_pred):.6f} ({correct} out of {test.n} correct)")
    print_confusion_matrix(matrix)
    if all(sum(row) > 0 for row in matrix):
        print(f"Balanced error rate: {balanced_error_rate(matrix):.6f}")


if __name__ == "__main__":
    main()
