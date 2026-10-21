# AIT626: Sentiment Classification with Naïve Bayes (Assignment 2)

## Project Overview
This project builds a Naïve Bayes sentiment classifier from scratch, using only foundational libraries like NumPy and NLTK for tokenization and stemming, without relying on pre-build machine learning classifiers like scikit-learn [cite: 1]. The model classifies airline review tweets into two sentiment classes: **Positive** and **Negative** [cite: 1].

## Team Members and Responsibilities
**Group 3**
- Nathania: Data preprocessing, vocabulary construction ($V$), and feature extraction, Bag of Words frequency and binary vector representation [cite: 1].
- Farzad: Naïve Bayes algorithm implementation, prior calculation, likelihood estimation, and test set predition.
- Yuze: Evaluation pipeline, confusion matrix generation, experimental variations (stemming versus no-stemming, bonus tasks), documentation, and code integration.

---

## Project Structure
'''text
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
