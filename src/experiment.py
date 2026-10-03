from pathlib import Path

import pandas as pd

from src.data_loader import load_dataset
from src.evaluation import evaluate_predictions
from src.features import (
    create_tfidf_vectorizer,
    fit_transform_training_data,
    transform_test_data,
)
from src.models import (
    create_logistic_regression_model,
    create_naive_bayes_model,
    train_model,
)
from src.preprocessing import preprocess_reviews


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Preprocess training and test review text.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset containing review text and labels.

    test_df : pandas.DataFrame
        Test dataset containing review text and labels.

    Returns
    -------
    tuple[list[str], list[str]]
        Preprocessed training and test texts.
    """
    train_texts = preprocess_reviews(train_df["text"])
    test_texts = preprocess_reviews(test_df["text"])

    return train_texts, test_texts


def select_training_subset(
    train_df: pd.DataFrame,
    fraction: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Select a reproducible subset of the training dataset.

    Stratification is applied through proportional sampling within
    each sentiment class so that the positive/negative balance is
    preserved.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Complete labelled training dataset.

    fraction : float
        Proportion of the training data to use.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Selected training subset.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and no greater than 1")

    subset_parts = []

    for label in sorted(train_df["label"].unique()):
        class_data = train_df[train_df["label"] == label]

        subset = class_data.sample(
            frac=fraction,
            random_state=random_state,
        )

        subset_parts.append(subset)

    subset_df = pd.concat(subset_parts)

    return subset_df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)


def run_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    training_fraction: float,
) -> dict:
    """
    Run one complete sentiment-classification experiment.

    The test dataset remains fixed and is never sampled or used
    during model/vectorizer fitting.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Complete training dataset.

    test_df : pandas.DataFrame
        Held-out test dataset.

    model_name : str
        Name of the classifier: 'naive_bayes' or
        'logistic_regression'.

    training_fraction : float
        Fraction of the training data used for model training.

    Returns
    -------
    dict
        Experiment configuration and evaluation results.
    """
    training_subset = select_training_subset(
        train_df,
        training_fraction,
    )

    train_texts = preprocess_reviews(training_subset["text"])
    test_texts = preprocess_reviews(test_df["text"])

    vectorizer = create_tfidf_vectorizer()

    X_train = fit_transform_training_data(
        vectorizer,
        train_texts,
    )

    X_test = transform_test_data(
        vectorizer,
        test_texts,
    )

    y_train = training_subset["label"]
    y_test = test_df["label"]

    if model_name == "naive_bayes":
        model = create_naive_bayes_model()
    elif model_name == "logistic_regression":
        model = create_logistic_regression_model()
    else:
        raise ValueError(
            "model_name must be 'naive_bayes' or 'logistic_regression'"
        )

    model = train_model(model, X_train, y_train)

    predictions = model.predict(X_test)

    evaluation = evaluate_predictions(
        y_test,
        predictions,
    )

    result = {
        "model": model_name,
        "training_fraction": training_fraction,
        "training_samples": len(training_subset),
        "test_samples": len(test_df),
        "vocabulary_size": len(vectorizer.vocabulary_),
        **evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"].tolist(),
    }

    return result


def save_result(result: dict):
    """
    Save one experiment result as a CSV file.

    Parameters
    ----------
    result : dict
        Experiment result dictionary.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{result['model']}_"
        f"{int(result['training_fraction'] * 100)}pct.csv"
    )

    output_path = RESULTS_DIR / filename

    result_to_save = result.copy()
    result_to_save["confusion_matrix"] = str(
        result_to_save["confusion_matrix"]
    )

    pd.DataFrame([result_to_save]).to_csv(
        output_path,
        index=False,
    )

    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    print("Loading IMDb dataset...")

    train_df, test_df = load_dataset()

    print(f"Training reviews available: {len(train_df):,}")
    print(f"Test reviews: {len(test_df):,}")

    print("\nRunning baseline experiment...")

    result = run_experiment(
        train_df=train_df,
        test_df=test_df,
        model_name="naive_bayes",
        training_fraction=0.10,
    )

    print("\nExperiment result:")
    for key, value in result.items():
        print(f"{key}: {value}")

    save_result(result)

    print("\nBaseline experiment completed.")