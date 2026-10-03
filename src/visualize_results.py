from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "all_experiments.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def plot_naive_bayes_training_effect(df: pd.DataFrame):
    """Plot Naive Bayes performance against training-data size."""

    nb = df[
        (df["model"] == "naive_bayes")
        & (df["preprocessing"] == "baseline")
    ].sort_values("training_fraction")

    x = nb["training_fraction"] * 100

    plt.figure(figsize=(9, 6))

    plt.plot(
        x,
        nb["accuracy"] * 100,
        marker="o",
        label="Accuracy",
    )

    plt.plot(
        x,
        nb["f1"] * 100,
        marker="o",
        label="F1 Score",
    )

    plt.xlabel("Training Data Used (%)")
    plt.ylabel("Performance (%)")
    plt.title("Naive Bayes Performance vs. Training Data Size")
    plt.xticks([10, 30, 60, 100])
    plt.ylim(75, 95)
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = FIGURES_DIR / "naive_bayes_training_effect.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_FILE)

    plot_naive_bayes_training_effect(df)

    print("Visualization completed.")