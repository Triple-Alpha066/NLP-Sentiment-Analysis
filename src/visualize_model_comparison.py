from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "all_experiments.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def plot_model_comparison(df: pd.DataFrame):
    """Compare baseline models trained on 100% of the training data."""

    comparison = df[
        (df["preprocessing"] == "baseline")
        & (df["training_fraction"] == 1.0)
        & (df["model"].isin(["naive_bayes", "logistic_regression"]))
    ].copy()

    comparison = comparison.sort_values("model")

    models = comparison["model"].replace(
        {
            "naive_bayes": "Naive Bayes",
            "logistic_regression": "Logistic Regression",
        }
    )

    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    x = range(len(metrics))
    width = 0.35

    nb_values = (
        comparison[comparison["model"] == "naive_bayes"][metrics]
        .iloc[0]
        .values
    )

    lr_values = (
        comparison[comparison["model"] == "logistic_regression"][metrics]
        .iloc[0]
        .values
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - width / 2 for i in x],
        nb_values * 100,
        width=width,
        label="Naive Bayes",
    )

    plt.bar(
        [i + width / 2 for i in x],
        lr_values * 100,
        width=width,
        label="Logistic Regression",
    )

    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score (%)")
    plt.title("Model Performance Comparison Using 100% Training Data")
    plt.xticks(list(x), metric_labels)
    plt.ylim(75, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    output_path = FIGURES_DIR / "model_comparison_100_percent.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_FILE)

    plot_model_comparison(df)

    print("Model comparison visualization completed.")