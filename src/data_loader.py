from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "aclImdb"


def load_reviews(split: str) -> pd.DataFrame:
    """
    Load labelled IMDb movie reviews from the specified dataset split.

    Parameters
    ----------
    split : str
        Dataset split to load. Must be either 'train' or 'test'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - text: the movie review
        - label: sentiment label (0 = negative, 1 = positive)
        - split: dataset split ('train' or 'test')
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'")

    records = []

    for sentiment, label in [("neg", 0), ("pos", 1)]:
        sentiment_dir = DATASET_DIR / split / sentiment

        if not sentiment_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {sentiment_dir}"
            )

        for review_file in sorted(sentiment_dir.glob("*.txt")):
            text = review_file.read_text(encoding="utf-8", errors="replace")

            records.append(
                {
                    "text": text,
                    "label": label,
                    "split": split,
                }
            )

    return pd.DataFrame(records)


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the labelled IMDb training and test datasets.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Training DataFrame and test DataFrame.
    """
    train_df = load_reviews("train")
    test_df = load_reviews("test")

    return train_df, test_df


if __name__ == "__main__":
    train_data, test_data = load_dataset()

    print("IMDb dataset loaded successfully.")
    print(f"Training reviews: {len(train_data):,}")
    print(f"Test reviews: {len(test_data):,}")

    print("\nTraining class distribution:")
    print(train_data["label"].value_counts().sort_index())

    print("\nTest class distribution:")
    print(test_data["label"].value_counts().sort_index())

    print("\nSample review:")
    print(train_data.iloc[0]["text"][:500])