# Methodology

## 1. Project Overview

This project investigates sentiment analysis of movie reviews using Natural Language Processing (NLP) and supervised machine learning. The objective is to develop a reproducible text-classification pipeline capable of assigning movie reviews to one of two sentiment classes: positive or negative.

The experimental design investigates three principal factors:

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

The official test set is kept completely separate from model training throughout all experiments.

---

## 3. Data Loading

The dataset is loaded from the extracted IMDb directory structure. Each review is represented as a record containing:

- `text`: the movie-review text.
- `label`: the sentiment class.
- `split`: the dataset split.

Sentiment labels are encoded numerically:

- `0` = negative.
- `1` = positive.

The data-loading implementation validates the requested dataset split and reads the review files using UTF-8 encoding with replacement for invalid characters.

---

## 4. Text Preprocessing

Two preprocessing configurations are evaluated: a baseline configuration and an enhanced configuration.

### 4.1 Baseline Preprocessing

The baseline preprocessing performs conservative text normalization designed to remove structural noise while preserving potentially sentiment-bearing linguistic information.

The baseline preprocessing steps are:

1. Removal of HTML tags.
2. Removal of URLs.
3. Conversion of text to lowercase.
4. Normalization of repeated whitespace.

Negation is deliberately preserved. Words such as `not`, `no`, and `never` are retained because they can alter the polarity of a sentence and therefore may contain important sentiment information.

### 4.2 Enhanced Preprocessing

The enhanced preprocessing configuration extends the baseline normalization with additional conventional NLP preprocessing operations.

The enhanced preprocessing steps are:

1. Removal of HTML tags.
2. Removal of URLs.
3. Conversion of text to lowercase.
4. Extraction of alphabetic tokens.
5. Removal of stop words.
6. Preservation of selected negation terms: `no`, `nor`, `not`, and `never`.
7. WordNet lemmatization.
8. Normalization of the resulting token sequence.

The implementation uses WordNet lemmatization rather than stemming. Lemmatization is applied without explicit part-of-speech tagging.

The enhanced preprocessing condition is evaluated experimentally against the baseline configuration to determine whether additional linguistic normalization improves classification performance.

---

## 5. Text Representation

The cleaned review text is converted into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).

The TF-IDF vectorizer uses:

- Unigrams and bigrams (`ngram_range=(1, 2)`).
- Minimum document frequency of 2 (`min_df=2`).
- Maximum document frequency of 95% (`max_df=0.95`).
- Sublinear term-frequency scaling (`sublinear_tf=True`).

The use of both unigrams and bigrams allows the representation to capture individual words as well as short expressions.

To prevent information leakage, the TF-IDF vectorizer is fitted exclusively on the training data for each experiment. The held-out test data is transformed using the already-fitted vectorizer and is never used to learn the vocabulary or inverse-document-frequency weights.

For the training-data experiment, the vectorizer is independently fitted on each selected training subset so that the vocabulary is derived only from the data available to that particular experiment.

---

## 6. Classification Models

Two supervised classification algorithms are evaluated.

### 6.1 Multinomial Naive Bayes

Multinomial Naive Bayes is used as a probabilistic baseline classifier for text classification. It provides a computationally efficient baseline for evaluating sentiment classification using sparse textual features.

The Multinomial Naive Bayes model is used for the progressive training-data experiment and for comparison with Logistic Regression using the complete training dataset.

### 6.2 Logistic Regression

Logistic Regression is used as a second supervised classifier. It provides a linear classification approach that can operate effectively with high-dimensional sparse TF-IDF representations.

The Logistic Regression implementation uses a maximum of 1,000 iterations and a fixed random state of 42 for reproducibility.

---

## 7. Training-Data Experiment

To investigate the effect of training-data volume, Multinomial Naive Bayes is evaluated using four progressively larger subsets of the labelled training data:

- 10% = 2,500 reviews.
- 30% = 7,500 reviews.
- 60% = 15,000 reviews.
- 100% = 25,000 reviews.

The positive and negative classes are sampled proportionally so that their balance is maintained across the training subsets.

The same 25,000-review official test set is used for every experiment.

A random state of 42 is used for reproducible subset selection.

The purpose of this experiment is to determine whether increasing the quantity of labelled training data improves classification performance and whether the magnitude of improvement changes as the training set becomes larger.

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

This controlled setup allows the performance of the two supervised algorithms to be compared under the same data and feature-representation conditions.

---

## 9. Preprocessing Comparison Experiment

A separate experiment evaluates whether enhanced preprocessing improves sentiment-classification performance.

The comparison uses:

- The complete 25,000-review training set.
- The same 25,000-review official test set.
- The same TF-IDF configuration.
- Logistic Regression as the classifier.
- The same evaluation metrics.

Two preprocessing conditions are compared:

1. Baseline preprocessing.
2. Enhanced preprocessing incorporating stop-word removal with selected negation terms preserved and WordNet lemmatization.

The purpose of this experiment is to isolate the effect of the preprocessing configuration while keeping the classifier, data splits and feature-extraction settings constant.

The baseline and enhanced configurations are evaluated using accuracy, precision, recall and F1-score. Vocabulary size is also recorded to determine the effect of enhanced preprocessing on the dimensionality of the TF-IDF representation.

---

## 10. Evaluation Metrics

Model performance is evaluated using:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.

Accuracy measures the proportion of correctly classified reviews.

Precision measures the proportion of reviews predicted as positive that are actually positive.

Recall measures the proportion of actual positive reviews correctly identified by the classifier.

F1-score provides a combined measure of precision and recall.

The confusion matrix is used to examine the distribution of true positives, true negatives, false positives and false negatives.

Using multiple metrics provides a broader assessment of classifier performance than accuracy alone.

---

## 11. Error Analysis

Qualitative error analysis is performed using the predictions generated by the Logistic Regression baseline model on the complete test set.

The error-analysis pipeline:

1. Generates predictions for all 25,000 test reviews.
2. Identifies incorrectly classified reviews.
3. Separates errors into false positives and false negatives.
4. Calculates the predicted probability associated with the positive class.
5. Records prediction confidence.
6. Saves the misclassified reviews and associated prediction information for inspection.

A sample of misclassified reviews is examined qualitatively to identify recurring linguistic patterns associated with classification errors.

The analysis focuses on categories including:

- Mixed or qualified sentiment.
- Context-dependent sentiment.
- Sarcasm, irony or rhetorical language.
- Competing positive and negative evidence within long reviews.
- Possible disagreement between the review text and the assigned dataset label.

These categories are used for qualitative interpretation rather than as mutually exclusive statistical classes.

---

## 12. Experimental Results

### 12.1 Training-Data Experiment

The first experimental series evaluates Multinomial Naive Bayes using progressively larger training subsets.

| Training Data | Training Samples | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| 10% | 2,500 | 84.80% | 89.56% | 78.78% | 83.83% |
| 30% | 7,500 | 86.51% | 90.23% | 81.88% | 85.85% |
| 60% | 15,000 | 86.87% | 90.01% | 82.95% | 86.34% |
| 100% | 25,000 | 87.22% | 90.26% | 83.44% | 86.71% |

Performance increased as more labelled training data was provided. However, the magnitude of improvement decreased at larger training sizes.

Accuracy increased by 2.42 percentage points between the 10% and 100% training conditions, while F1-score increased by 2.88 percentage points.

---

### 12.2 Model Comparison

Using the complete 25,000-review training set and the baseline preprocessing configuration, the two classifiers achieved the following results:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 87.22% | 90.26% | 83.44% | 86.71% |
| Logistic Regression | 89.66% | 89.40% | 89.98% | 89.69% |

Logistic Regression achieved higher accuracy, recall and F1-score, while Multinomial Naive Bayes achieved slightly higher precision.

---

### 12.3 Preprocessing Comparison

The preprocessing experiment compared Logistic Regression using baseline and enhanced preprocessing configurations.

| Preprocessing | Accuracy | Precision | Recall | F1-score | Vocabulary Size |
|---|---:|---:|---:|---:|---:|
| Baseline | 89.66% | 89.40% | 89.98% | 89.69% | 436,525 |
| Enhanced | 89.15% | 88.72% | 89.70% | 89.21% | 350,099 |

Enhanced preprocessing reduced the TF-IDF vocabulary from 436,525 to 350,099 features, representing an approximate 19.8% reduction in vocabulary size.

However, the enhanced configuration produced a small reduction in all four classification metrics. Under the evaluated configuration, vocabulary reduction therefore did not translate into improved predictive performance.

---

## 13. Reproducibility

Reproducibility is supported through:

- A fixed random state of 42 for training-subset selection.
- A fixed random state of 42 for Logistic Regression.
- Explicit TF-IDF parameters.
- Separate Python modules for data loading, preprocessing, feature extraction, model training and evaluation.
- CSV storage of experimental results.
- A documented project structure.
- A `requirements.txt` file specifying the Python dependencies.
- Version-controlled source code maintained in the project Git repository.

The experimental configuration is documented alongside the implementation so that the reported results can be reproduced from the project repository.

---

## 14. Experimental Limitations

Several limitations should be considered when interpreting the results.

First, the experiments evaluate only two conventional supervised classifiers: Multinomial Naive Bayes and Logistic Regression. Other algorithms and modern neural or transformer-based approaches may produce different results.

Second, the preprocessing comparison evaluates one enhanced preprocessing configuration. Therefore, the observed reduction in performance cannot be interpreted as evidence that stop-word removal or lemmatization is universally detrimental to sentiment classification.

Third, the training-data volume experiment is performed using Multinomial Naive Bayes. Consequently, the observed diminishing gains describe the evaluated Naive Bayes configuration and should not automatically be generalized to all classifiers.

Finally, qualitative error analysis is based on inspection of sampled misclassified reviews. The identified error categories provide interpretive insight but are not intended to represent statistically estimated proportions of the complete error population.

---

## 15. Summary of Experimental Design

The complete experimental workflow can be summarized as follows:

1. Load the labelled IMDb movie-review dataset.
2. Keep the official test set separate from model training.
3. Apply baseline text preprocessing.
4. Convert reviews into TF-IDF unigram and bigram features.
5. Evaluate Multinomial Naive Bayes using progressively larger training subsets.
6. Compare Multinomial Naive Bayes and Logistic Regression using the complete training set.
7. Compare baseline and enhanced preprocessing using Logistic Regression.
8. Evaluate all configurations using accuracy, precision, recall, F1-score and confusion matrices.
9. Analyse Logistic Regression misclassifications qualitatively.
10. Store experimental results and visualizations in the project repository.

This methodology provides a controlled and reproducible framework for investigating the effects of training-data volume, classifier selection and preprocessing configuration on binary sentiment classification of IMDb movie reviews.