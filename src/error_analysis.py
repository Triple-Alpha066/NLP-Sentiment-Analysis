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
    train_model,
)
from src.preprocessing import preprocess_reviews


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "predictions"


def run_error_analysis():
    """Train the baseline Logistic Regression model and save misclassified reviews."""

    print("Loading IMDb dataset...")
    train_df, test_df = load_dataset()

    print(f"Training reviews: {len(train_df):,}")
    print(f"Test reviews: {len(test_df):,}")

    # ---------------------------------------------------------
    # 1. Preprocess training and test reviews
    # ---------------------------------------------------------
    print("\nPreprocessing reviews...")

    train_texts = preprocess_reviews(train_df["text"])
    test_texts = preprocess_reviews(test_df["text"])

    # ---------------------------------------------------------
    # 2. Fit TF-IDF ONLY on training data
    # ---------------------------------------------------------
    print("Creating TF-IDF representation...")

    vectorizer = create_tfidf_vectorizer()

    X_train = fit_transform_training_data(
        vectorizer,
        train_texts,
    )

    X_test = transform_test_data(
        vectorizer,
        test_texts,
    )

    # ---------------------------------------------------------
    # 3. Train Logistic Regression
    # ---------------------------------------------------------
    print("Training Logistic Regression...")

    y_train = train_df["label"]
    y_test = test_df["label"]

    model = create_logistic_regression_model()
    model = train_model(model, X_train, y_train)

    # ---------------------------------------------------------
    # 4. Generate predictions and probabilities
    # ---------------------------------------------------------
    print("Generating predictions...")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    positive_probability = probabilities[:, 1]

    # Confidence is the probability assigned to the predicted class.
    confidence = probabilities.max(axis=1)

    # ---------------------------------------------------------
    # 5. Evaluate the complete test set
    # ---------------------------------------------------------
    evaluation = evaluate_predictions(
        y_test,
        predictions,
    )

    print("\nOverall model performance:")
    for metric, value in evaluation["metrics"].items():
        print(f"{metric}: {value:.4f}")

    # ---------------------------------------------------------
    # 6. Identify misclassified reviews
    # ---------------------------------------------------------
    errors = predictions != y_test.to_numpy()

    misclassified = test_df.loc[errors].copy()

    misclassified["true_label"] = y_test.to_numpy()[errors]
    misclassified["predicted_label"] = predictions[errors]
    misclassified["positive_probability"] = positive_probability[errors]
    misclassified["confidence"] = confidence[errors]

    # Keep the original test-set position for traceability.
    misclassified["test_index"] = misclassified.index

    # ---------------------------------------------------------
    # 7. Add interpretable error direction
    # ---------------------------------------------------------
    misclassified["error_type"] = misclassified.apply(
        lambda row: (
            "False Positive"
            if row["true_label"] == 0 and row["predicted_label"] == 1
            else "False Negative"
        ),
        axis=1,
    )

    # ---------------------------------------------------------
    # 8. Select and order columns
    # ---------------------------------------------------------
    misclassified = misclassified[
        [
            "test_index",
            "text",
            "true_label",
            "predicted_label",
            "error_type",
            "positive_probability",
            "confidence",
        ]
    ]

    # Sort by confidence so that highly confident mistakes
    # can be examined during qualitative error analysis.
    misclassified = misclassified.sort_values(
        by="confidence",
        ascending=False,
    )

    # ---------------------------------------------------------
    # 9. Save results
    # ---------------------------------------------------------
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "logistic_regression_misclassifications.csv"
    )

    misclassified.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 10. Print summary
    # ---------------------------------------------------------
    false_positives = (
        (misclassified["error_type"] == "False Positive").sum()
    )

    false_negatives = (
        (misclassified["error_type"] == "False Negative").sum()
    )

    print("\nError-analysis summary:")
    print(f"Total test reviews: {len(test_df):,}")
    print(f"Correct predictions: {(~errors).sum():,}")
    print(f"Misclassified reviews: {errors.sum():,}")
    print(f"False positives: {false_positives:,}")
    print(f"False negatives: {false_negatives:,}")

    print(f"\nSaved misclassified reviews to:")
    print(output_path)


if __name__ == "__main__":
    run_error_analysis()