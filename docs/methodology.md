# Methodology

## 1. Project Overview

This project investigates sentiment analysis of movie reviews using Natural Language Processing (NLP) and supervised machine learning. The objective is to develop a reproducible text-classification pipeline capable of assigning movie reviews to one of two sentiment classes: positive or negative.

The experimental design focuses on three factors:

1. The amount of labelled training data available to the classifier.
2. The choice of supervised classification algorithm.
3. The effect of text preprocessing on classification performance.

The primary evaluation is performed on a fixed held-out test dataset so that results from different experiments remain directly comparable.

---

## 2. Dataset

The project uses the IMDb Large Movie Review Dataset, which contains labelled movie reviews for binary sentiment classification.

The dataset contains:

- 25,000 labelled training reviews.
- 25,000 labelled test reviews.
- 12,500 positive and 12,500 negative reviews in each labelled split.

The dataset also contains an `unsup` directory within the training data. These unlabelled reviews are excluded because the current project focuses on supervised binary classification.

The official test set is kept separate from model training throughout the experiments.

---

## 3. Data Loading

The dataset is loaded from the extracted IMDb directory structure. Each review is represented as a record containing:

- `text`: the movie-review text.
- `label`: the sentiment class.
- `split`: the dataset split.

Sentiment labels are encoded numerically:

- `0` = negative
- `1` = positive

The data-loading implementation validates the requested dataset split and reads the review files using UTF-8 encoding with replacement for invalid characters.

---

## 4. Preprocessing

The initial experimental pipeline uses conservative text normalization designed to remove structural noise while preserving potentially sentiment-bearing linguistic information.

The baseline preprocessing steps are:

1. Removal of HTML tags.
2. Removal of URLs.
3. Conversion of text to lowercase.
4. Normalization of repeated whitespace.

Negation is deliberately preserved. Words such as `not`, `no`, and `never` can alter the polarity of a sentence and therefore are not blindly removed during baseline preprocessing.

This baseline will subsequently be compared with a second preprocessing condition incorporating additional conventional NLP preprocessing operations required by the project specification.

---

## 5. Text Representation

The cleaned review text is converted into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).

The baseline vectorizer uses:

- Unigrams and bigrams (`ngram_range=(1, 2)`).
- Minimum document frequency of 2.
- Maximum document frequency of 95%.
- Sublinear term-frequency scaling.

The use of both unigrams and bigrams allows the representation to capture individual words as well as short expressions.

To prevent information leakage, the TF-IDF vectorizer is fitted exclusively on the training data for each experiment. The held-out test data is transformed using the already-fitted vectorizer and is never used to learn the vocabulary or inverse-document-frequency weights.

---

## 6. Classification Models

Two supervised classification algorithms are evaluated.

### 6.1 Multinomial Naive Bayes

Multinomial Naive Bayes is used as a probabilistic baseline classifier for text classification. It provides a computationally efficient baseline for evaluating sentiment classification using sparse textual features.

### 6.2 Logistic Regression

Logistic Regression is used as a second supervised classifier. It provides a linear classification approach that can operate effectively with high-dimensional sparse TF-IDF representations.

The Logistic Regression implementation uses a maximum of 1,000 iterations and a fixed random state of 42 for reproducibility.

---

## 7. Training-Data Experiment

To investigate the effect of training-data volume, Multinomial Naive Bayes is evaluated using four progressively larger subsets of the labelled training data:

- 10% = 2,500 reviews
- 30% = 7,500 reviews
- 60% = 15,000 reviews
- 100% = 25,000 reviews

The positive and negative classes are sampled proportionally so that their balance is maintained across the training subsets.

The same 25,000-review test set is used for every experiment.

A random state of 42 is used for reproducible subset selection.

---

## 8. Model Comparison Experiment

After completing the progressive training-data experiment, Multinomial Naive Bayes and Logistic Regression are compared using the complete 25,000-review training set.

The following experimental conditions are held constant:

- Dataset.
- Training set.
- Test set.
- Baseline preprocessing.
- TF-IDF configuration.
- Evaluation metrics.

The classifier is therefore the principal experimental variable in this comparison.

---

## 9. Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Accuracy measures the proportion of correctly classified reviews.

Precision measures the proportion of reviews predicted as positive that are actually positive.

Recall measures the proportion of actual positive reviews correctly identified by the classifier.

F1-score provides a combined measure of precision and recall.

The confusion matrix is used to examine the distribution of true positives, true negatives, false positives, and false negatives.

---

## 10. Reproducibility

Reproducibility is supported through:

- A fixed random state of 42 for dataset subset selection.
- A fixed random state of 42 for Logistic Regression.
- Explicit TF-IDF parameters.
- Separate Python modules for data loading, preprocessing, feature extraction, model training, and evaluation.
- CSV storage of experimental results.

The experimental configuration is documented alongside the implementation so that the reported results can be reproduced from the project repository.

---

## 11. Current Experimental Results

The first experimental series evaluates Multinomial Naive Bayes using progressively larger training subsets.

| Training Data | Training Samples | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| 10% | 2,500 | 84.80% | 89.56% | 78.78% | 83.83% |
| 30% | 7,500 | 86.51% | 90.23% | 81.88% | 85.85% |
| 60% | 15,000 | 86.87% | 90.01% | 82.95% | 86.34% |
| 100% | 25,000 | 87.22% | 90.26% | 83.44% | 86.71% |

The results show increasing performance as more labelled training data is provided, although the magnitude of improvement decreases at larger training sizes.

Using the complete training set, Logistic Regression achieved:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 87.22% | 90.26% | 83.44% | 86.71% |
| Logistic Regression | 89.66% | 89.40% | 89.98% | 89.69% |

These results are preliminary experimental findings and will be extended through the preprocessing experiment and subsequent error analysis.

---

## 12. Planned Preprocessing Experiment

A second preprocessing condition will be implemented to investigate the effect of additional conventional NLP preprocessing.

The experiment will compare the baseline preprocessing condition against a preprocessing condition incorporating stop-word removal and either stemming or lemmatization.

The same classifier, training data, test data, feature representation, and evaluation procedure will be maintained where possible so that the preprocessing strategy remains the principal experimental variable.

The results will be used to assess whether additional preprocessing improves sentiment-classification performance on the IMDb dataset.