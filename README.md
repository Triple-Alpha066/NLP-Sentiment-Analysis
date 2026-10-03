# Sentiment Analysis of Movie Reviews Using NLP and Machine Learning

## Project Overview

This project develops and evaluates a Natural Language Processing (NLP) pipeline for binary sentiment classification of movie reviews.

The system classifies movie reviews as either:

- **Positive**
- **Negative**

The project investigates the effects of:

1. Training-data volume
2. Classification algorithm
3. Text preprocessing

The experimental study uses the IMDb Large Movie Review Dataset and compares Multinomial Naive Bayes and Logistic Regression using TF-IDF text representations.

---

## Research Questions

The project addresses four research questions:

1. How does text preprocessing affect sentiment-classification performance on movie reviews?
2. How does the choice of supervised classification algorithm affect performance when the same textual representation and test set are used?
3. How does increasing the amount of training data affect classification performance?
4. What types of movie reviews are most difficult for the classifiers to classify correctly?

---

## Dataset

The project uses the **IMDb Large Movie Review Dataset**.

The labelled dataset contains:

- 25,000 training reviews
  - 12,500 positive
  - 12,500 negative
- 25,000 test reviews
  - 12,500 positive
  - 12,500 negative

The unlabelled training reviews were excluded from the supervised experiments.

The official test set is kept separate from model training and is used for final evaluation.

The raw dataset is intentionally excluded from this repository through `.gitignore`.

---

## NLP Pipeline

The implemented pipeline follows this structure:

```text
Raw Movie Reviews
        ↓
Data Loading
        ↓
Text Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
Supervised Classification
        ↓
Prediction
        ↓
Evaluation
        ↓
Error Analysis