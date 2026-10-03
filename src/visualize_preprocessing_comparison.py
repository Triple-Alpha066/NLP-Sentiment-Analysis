from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "all_experiments.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def plot_preprocessing_comparison(df: pd.DataFrame):
    """Compare baseline and enhanced preprocessing for Logistic Regression."""

    comparison = df[
        (df["model"] == "logistic_regression")
        & (df["training_fraction"] == 1.0)
        & (df["preprocessing"].isin(["baseline", "enhanced"]))
    ].copy()

    preprocessing_labels = comparison["preprocessing"].replace(
        {
            "baseline": "Baseline",
            "enhanced": "Enhanced",
        }
    )

    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    values = comparison[metrics].values

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - width / 2 for i in x],
        values[0] * 100,
        width=width,
        label=preprocessing_labels.iloc[0],
    )

    plt.bar(
        [i + width / 2 for i in x],
        values[1] * 100,
        width=width,
        label=preprocessing_labels.iloc[1],
    )

    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score (%)")
    plt.title(
        "Effect of Preprocessing on Logistic Regression Performance"
    )
    plt.xticks(list(x), metric_labels)
    plt.ylim(75, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    output_path = FIGURES_DIR / "preprocessing_comparison_logistic_regression.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_FILE)

    plot_preprocessing_comparison(df)

    print("Preprocessing comparison visualization completed.")