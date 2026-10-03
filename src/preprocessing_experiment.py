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
from src.preprocessing_enhanced import preprocess_reviews_enhanced


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"


def run_preprocessing_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessing_type: str,
    model_name: str,
) -> dict:
    """
    Compare preprocessing strategies under a fixed model configuration.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Complete labelled training dataset.

    test_df : pandas.DataFrame
        Held-out labelled test dataset.

    preprocessing_type : str
        Either 'baseline' or 'enhanced'.

    model_name : str
        Either 'naive_bayes' or 'logistic_regression'.

    Returns
    -------
    dict
        Experiment configuration and evaluation results.
    """

    if preprocessing_type == "baseline":
        preprocess_function = preprocess_reviews
    elif preprocessing_type == "enhanced":
        preprocess_function = preprocess_reviews_enhanced
    else:
        raise ValueError(
            "preprocessing_type must be 'baseline' or 'enhanced'"
        )

    train_texts = preprocess_function(train_df["text"])
    test_texts = preprocess_function(test_df["text"])

    vectorizer = create_tfidf_vectorizer()

    X_train = fit_transform_training_data(
        vectorizer,
        train_texts,
    )

    X_test = transform_test_data(
        vectorizer,
        test_texts,
    )

    y_train = train_df["label"]
    y_test = test_df["label"]

    if model_name == "naive_bayes":
        model = create_naive_bayes_model()
    elif model_name == "logistic_regression":
        model = create_logistic_regression_model()
    else:
        raise ValueError(
            "model_name must be 'naive_bayes' or "
            "'logistic_regression'"
        )

    model = train_model(
        model,
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    evaluation = evaluate_predictions(
        y_test,
        predictions,
    )

    return {
        "preprocessing": preprocessing_type,
        "model": model_name,
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "vocabulary_size": len(vectorizer.vocabulary_),
        **evaluation["metrics"],
        "confusion_matrix": (
            evaluation["confusion_matrix"].tolist()
        ),
    }


def save_preprocessing_result(result: dict):
    """
    Save a preprocessing experiment result as a CSV file.
    """
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"preprocessing_"
        f"{result['preprocessing']}_"
        f"{result['model']}.csv"
    )

    output_path = RESULTS_DIR / filename

    result_to_save = result.copy()

    result_to_save["confusion_matrix"] = str(
        result_to_save["confusion_matrix"]
    )

    pd.DataFrame(
        [result_to_save]
    ).to_csv(
        output_path,
        index=False,
    )

    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    print("Loading IMDb dataset...")

    train_df, test_df = load_dataset()

    print(f"Training reviews: {len(train_df):,}")
    print(f"Test reviews: {len(test_df):,}")

    print("\nRunning enhanced preprocessing experiment...")

    result = run_preprocessing_experiment(
        train_df=train_df,
        test_df=test_df,
        preprocessing_type="enhanced",
        model_name="logistic_regression",
    )

    print("\nExperiment result:")

    for key, value in result.items():
        print(f"{key}: {value}")

    save_preprocessing_result(result)

    print("\nPreprocessing experiment completed.")