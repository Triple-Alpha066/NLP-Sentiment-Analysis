import re


def clean_text(text: str) -> str:
    """
    Normalize a movie review while preserving sentiment-bearing words.

    The preprocessing steps:
    1. Remove HTML tags such as <br />
    2. Replace URLs with a space
    3. Convert text to lowercase
    4. Normalize repeated whitespace

    Parameters
    ----------
    text : str
        Raw movie review text.

    Returns
    -------
    str
        Cleaned review text.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Convert to lowercase.
    text = text.lower()

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_reviews(texts):
    """
    Apply clean_text() to a collection of reviews.

    Parameters
    ----------
    texts : iterable
        Collection of raw review texts.

    Returns
    -------
    list[str]
        Cleaned review texts.
    """
    return [clean_text(text) for text in texts]


if __name__ == "__main__":
    sample = (
        "<br /><br />This movie was NOT boring! "
        "Visit https://example.com for more information."
    )

    cleaned = clean_text(sample)

    print("Original:")
    print(sample)

    print("\nCleaned:")
    print(cleaned)