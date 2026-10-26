# AIT626: Sentiment Classification with Naïve Bayes (Assignment 2)

## Project Overview
This project builds a Naïve Bayes sentiment classifier from scratch, using only foundational libraries like NumPy and NLTK for tokenization and stemming, without relying on pre-build machine learning classifiers like scikit-learn [1]. The model classifies airline review tweets into two sentiment classes: **Positive** and **Negative** [1].

## Team Members and Responsibilities
**Group 3**
- Nathania: Data preprocessing, vocabulary construction ($V$), and feature extraction, Bag of Words frequency and binary vector representation [cite: 1].
- Farzad: Naïve Bayes algorithm implementation, prior calculation, likelihood estimation, and test set precdition.
- Yuze: Evaluation pipeline, confusion matrix generation, experimental variations (stemming versus no-stemming, bonus tasks), documentation, and code integration.

---

## Project Structure
```text
AIT626_NaiveBayes_Sentiment/
│
├── data/                    # Training and test dataset splits (pos/neg directories)
├── src/                     # Source code files
│   ├── preprocessing.py     # Tokenization, vocab creation, BoW & binary feature extractors
│   ├── model.py             # Naïve Bayes training and prediction logic
│   └── evaluation.py        # Accuracy, confusion matrix, and performance metrics
├── outputs/                 # Saved results and log files (.txt / .log)
├── README.md                # Project documentation
└── main.py                  # End-to-end integration script
```
## Dataset Setup
1. Download the tweet dataset zip file provided for Assignment 2 in Canvas.
2. Extract the archive into the root directory so that the folder structure is as follows:
    ```text
   AIT626_NaiveBayes_Sentiment/
   ├── data/
   │   ├── train/
   │   │   ├── positive/
   │   │   └── negative/
   │   └── test/
   │       ├── positive/
   │       └── negative/
   ```
## Installation and Usage

Install the required dependencies:

```bash
python -m pip install nltk beautifulsoup4
```

Run the four required experiments:

```bash
python main.py --data-dir data --output-dir outputs
```

Run all six experiments, including the optional TF-IDF bonus:

```bash
python main.py --data-dir data --output-dir outputs --bonus
```

Run the evaluation unit tests:

```bash
python -m unittest discover -s tests -v
```

Results are saved in `outputs/results.txt`.

## Algorithm Description

The sentiment classifier was implemented from scratch using Multinomial Naïve Bayes.

1. **Preprocessing:** Remove HTML markup, tokenize reviews, apply capitalization rules, and optionally perform Porter stemming.
2. **Vocabulary Construction:** Build the vocabulary exclusively from the training data.
3. **Feature Extraction:** Represent each review using either word-frequency counts or binary word-presence features.
4. **Training:** Estimate class prior probabilities and word likelihoods with Laplace smoothing (α = 1).
5. **Prediction:** Compute class scores in log space and assign the class with the highest score.
6. **Evaluation:** Calculate classification accuracy, confusion matrices, precision, recall, and F1 scores.

Binary features are evaluated using binary-weighted Multinomial Naïve Bayes rather than a Bernoulli Naïve Bayes likelihood.

## Experimental Results

The experiments evaluated 4,182 test reviews.

| Configuration | Accuracy | Correct Predictions |
|---|---:|---:|
| No Stemming + Count | 89.05% | 3,724 |
| No Stemming + Binary | 88.59% | 3,705 |
| Stemming + Count | **89.60%** | **3,747** |
| Stemming + Binary | 89.50% | 3,743 |
| No Stemming + TF-IDF (Bonus) | 88.28% | 3,692 |
| Stemming + TF-IDF (Bonus) | 88.16% | 3,687 |

### Best Model: Stemming + Count

The highest accuracy was achieved by the Stemming + Count configuration, with 89.60% accuracy.

Confusion matrix (rows = actual labels, columns = predicted labels):

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 2,910 | 90 |
| Actual Positive | 345 | 837 |

### Discussion

Stemming improved classification accuracy for both the count-based and binary feature representations. It also reduced vocabulary size from 8,163 to 6,482 unique tokens.

Count-based features slightly outperformed binary features in both stemming configurations, suggesting that word frequency contributed useful information to sentiment classification.

The TF-IDF experiments achieved lower overall accuracy than the corresponding count-based models. However, they improved positive-class recall while increasing false-positive predictions, highlighting a trade-off between detecting positive reviews and overall classification accuracy.

These observations describe the results of the current dataset and implementation; no statistical significance test was conducted.

## Bonus Experiment: TF-IDF

**Bonus question completed.**

Two additional experiments were performed using term frequency multiplied by smoothed inverse document frequency, with IDF calculated from training documents only.

The resulting weights were used with the existing Multinomial Naïve Bayes classifier as an exploratory weighted-feature extension, rather than a conventional count-based probabilistic model.

## Input and Output Example

Example command:

```bash
python main.py --data-dir data --output-dir outputs --bonus
```

Example output:

```text
stem_count: accuracy=0.8960, cm=[[2910, 90], [345, 837]]
```

The program writes detailed results for all six experiments to `outputs/results.txt`, including accuracy, confusion matrices, and class-specific evaluation metrics.

## References
[1] Liao, D., PhD. (2026). <i>NLP Programming Assignment: Sentiment Classification with Naïve Bayes.</i> AIT 626 Natural Language Processing. George Mason University.<br>
[2] Kashetti, R., & Liao D., PhD. (2026). <i>Sentiment Classification with Naïve Bayes: Tips and Hints</i> AIT 626 Natural Language Processing. George Mason University.
