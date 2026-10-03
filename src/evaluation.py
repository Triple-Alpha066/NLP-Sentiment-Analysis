from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculate standard binary-classification performance metrics.

    Parameters
    ----------
    y_true : array-like
        True sentiment labels.

    y_pred : array-like
        Predicted sentiment labels.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, and F1-score.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix for binary sentiment classification.

    Parameters
    ----------
    y_true : array-like
        True sentiment labels.

    y_pred : array-like
        Predicted sentiment labels.

    Returns
    -------
    numpy.ndarray
        2x2 confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def evaluate_predictions(y_true, y_pred) -> dict:
    """
    Evaluate predictions using both performance metrics and a confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True sentiment labels.

    y_pred : array-like
        Predicted sentiment labels.

    Returns
    -------
    dict
        Evaluation results containing classification metrics and
        the confusion matrix.
    """
    metrics = calculate_metrics(y_true, y_pred)
    matrix = calculate_confusion_matrix(y_true, y_pred)

    return {
        "metrics": metrics,
        "confusion_matrix": matrix,
    }


if __name__ == "__main__":
    # Small controlled example for module testing.
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 0]

    results = evaluate_predictions(y_true, y_pred)

    print("Evaluation module test passed.")

    print("\nMetrics:")
    for metric, value in results["metrics"].items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion matrix:")
    print(results["confusion_matrix"])