from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create and configure the TF-IDF vectorizer.

    The vectorizer uses unigrams and bigrams so that the model can
    capture both individual words and short two-word expressions.

    Returns
    -------
    TfidfVectorizer
        Configured TF-IDF vectorizer.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def fit_transform_training_data(
    vectorizer: TfidfVectorizer,
    training_texts,
):
    """
    Fit the TF-IDF vectorizer on training data and transform it.

    The vectorizer must be fitted only on training data to prevent
    information from the test set from influencing the learned
    vocabulary or IDF weights.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        TF-IDF vectorizer to fit.

    training_texts : iterable
        Preprocessed training reviews.

    Returns
    -------
    scipy.sparse matrix
        TF-IDF representation of the training reviews.
    """
    return vectorizer.fit_transform(training_texts)


def transform_test_data(
    vectorizer: TfidfVectorizer,
    test_texts,
):
    """
    Transform test data using an already-fitted TF-IDF vectorizer.

    The vectorizer is not fitted again on the test data.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        TF-IDF vectorizer already fitted on training data.

    test_texts : iterable
        Preprocessed test reviews.

    Returns
    -------
    scipy.sparse matrix
        TF-IDF representation of the test reviews.
    """
    return vectorizer.transform(test_texts)


if __name__ == "__main__":
    sample_training_texts = [
        "This movie was excellent and entertaining.",
        "The acting was excellent.",
        "This movie was boring and disappointing.",
    ]

    sample_test_texts = [
        "The movie was entertaining.",
        "The acting was disappointing.",
    ]

    vectorizer = create_tfidf_vectorizer()

    X_train = fit_transform_training_data(
        vectorizer,
        sample_training_texts,
    )

    X_test = transform_test_data(
        vectorizer,
        sample_test_texts,
    )

    print("TF-IDF feature extraction test passed.")
    print(f"Training matrix shape: {X_train.shape}")
    print(f"Test matrix shape: {X_test.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")