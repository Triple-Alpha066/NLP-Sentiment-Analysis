from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def create_naive_bayes_model() -> MultinomialNB:
    """
    Create a Multinomial Naive Bayes classifier.

    Returns
    -------
    MultinomialNB
        Configured Naive Bayes classifier.
    """
    return MultinomialNB()


def create_logistic_regression_model() -> LogisticRegression:
    """
    Create a Logistic Regression classifier.

    The random_state value ensures reproducible results when
    the model is trained repeatedly under the same conditions.

    Returns
    -------
    LogisticRegression
        Configured Logistic Regression classifier.
    """
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
    )


def train_model(model, X_train, y_train):
    """
    Train a classifier using the training feature matrix and labels.

    Parameters
    ----------
    model : classifier
        A scikit-learn classification model.

    X_train : sparse matrix
        Training feature matrix.

    y_train : array-like
        Training labels.

    Returns
    -------
    classifier
        The fitted model.
    """
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    from scipy.sparse import csr_matrix

    X_sample = csr_matrix(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 0],
        ]
    )

    y_sample = [1, 0, 1, 0]

    naive_bayes = create_naive_bayes_model()
    logistic_regression = create_logistic_regression_model()

    train_model(naive_bayes, X_sample, y_sample)
    train_model(logistic_regression, X_sample, y_sample)

    print("Model training test passed.")
    print(
        "Naive Bayes classes:",
        naive_bayes.classes_,
    )
    print(
        "Logistic Regression classes:",
        logistic_regression.classes_,
    )