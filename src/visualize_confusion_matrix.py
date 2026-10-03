from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "all_experiments.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def plot_confusion_matrix(
    df: pd.DataFrame,
    model_name: str,
    preprocessing: str,
    output_name: str,
):
    """Plot a confusion matrix for a selected experiment."""

    result = df[
        (df["model"] == model_name)
        & (df["preprocessing"] == preprocessing)
        & (df["training_fraction"] == 1.0)
    ]

    if result.empty:
        raise ValueError("No matching experiment found.")

    matrix = eval(result.iloc[0]["confusion_matrix"])

    plt.figure(figsize=(7, 6))

    plt.imshow(matrix)

    plt.title(
        f"Confusion Matrix: "
        f"{model_name.replace('_', ' ').title()}"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{matrix[i][j]:,}",
                ha="center",
                va="center",
            )

    plt.colorbar(label="Number of Reviews")

    plt.tight_layout()

    output_path = FIGURES_DIR / output_name
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_FILE)

    plot_confusion_matrix(
        df=df,
        model_name="logistic_regression",
        preprocessing="baseline",
        output_name="confusion_matrix_logistic_regression.png",
    )

    print("Confusion matrix visualization completed.")