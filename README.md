## CS260 HW05: Naive Bayes

Name: Juliette Ortiz

---

### Implementation and testing

Use Python 3.9 or newer. No third-party packages are required.

`Partition.py` supplies shared typed data structures. `run_NB.py` also imports
`Example` and `Partition` for compatibility. Keep all required signatures and
type annotations. Complete the `NotImplementedError` placeholders; the command-
line parser, format dispatch, printing, and `main()` are provided.

- Readers return `Partition` objects, with string feature values and integer labels.
- `NaiveBayes.log_class[k]` stores a natural-log prior.
- `NaiveBayes.log_feature[feature][value][k]` stores a natural-log likelihood.
- `predict_log_scores()` returns unnormalized scores in class order.
- `classify()` returns an integer, choosing the smallest label in a tie.
- `predict()` returns a list of integer labels in input order.
- `confusion_matrix()` returns a list of lists of integer counts (actual rows).
- `accuracy()` and `balanced_error_rate()` return floats.

Computational helpers must not print, mutate inputs, or depend on command-line
state. Read the docstrings for edge-case behavior. Use training domains only;
missing features or out-of-domain test values raise `KeyError`. All supplied
census test values occur in their corresponding training domains.

Run the sample public tests with `python3 tests.py`. They initially fail until
you implement the required methods. Passing these tests does not guarantee
full credit. Run the full experiments as well, for example:

```bash
python3 run_NB.py -r data/zoo_train.arff -e data/zoo_test.arff
```

Include `Partition.py`, `NaiveBayes.py`, `run_NB.py`, and this completed README
in your submission, along with any additional helper modules.

### Written component

Complete `HW05_written_problems.pdf` by hand, or typeset your answers using
`HW05_written_problems.tex`. Show your work; a calculator is allowed, but do
not use programming to solve these problems. Submit a legible completed PDF
named `HW05_written_problems.pdf` alongside your code and this README. The
worksheet is separate from the analysis questions below.

### Analysis

#### Part 2

1. In this homework we will be considering sex as a protected attribute, and seeing if we can predict sex from the remaining attributes. First, compute the fractions of males and females in the *training* data.

Male 19633/28998  -> 68%
Female 9365/28998  -> 32%

2. If an algorithm to predict sex independently flipped a fair coin for every *test* example, what expected confusion matrix and expected accuracy would result? Use expectations rather than a single random simulation.

Should be write about 50% of the time, random guessing. 
Assuming Male is 0 and Female is 1

TN -> 9816.5 examples
FN -> 9816.5 examples
TP -> 4682.5 examples
FP -> 4682.5 examples


3. What if an algorithm always predicted the majority class from the *training* data? What confusion matrix and accuracy would it obtain on the test data?

Assuming Male is 0 and Female is 1

TN -> 19633
FN -> 0
TP -> 0
FP -> 9365

#### Part 4

1. Temporarily remove the Laplace counts and run the original census experiment again. What error, warning, or non-finite log value do you obtain? What does it mean about the data? Restore smoothing for the remaining experiments and your submission.

2. When trying to predict sex from the other attributes, what accuracy do you obtain? Is this higher or lower than you would expect with one of the naive strategies from Part 2?

3. Re-run your method with the `corrected` train/test data. What accuracy do you obtain now? Explain the difference between the two datasets and why this accuracy change makes sense.

4. Compute the balanced error rate (BER) for your method on the corrected census data. What does this value imply?

5. What (if anything) is concerning about being able to predict such a protected attribute
from the other attributes (especially using an algorithm like Naive Bayes that does not explicitly account for feature interactions)? If the actual "label" were "hired" or "not hired" for a job, and you
were responsible for making this decision, how would you deal with data where features were
redundantly encoded?

---

### Homework Questionnaire

(None of your answers below will affect your grade; this is to help refine homework
assignments in the future)

1. Approximately, how many hours did you take to complete this homework? (provide
  your answer as a single integer on the line below)

2. How difficult did you find this homework? (1-5, with 5 being very difficult and 1
  being very easy)

3. Describe the biggest challenge you faced on this homework:
