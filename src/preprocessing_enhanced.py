import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load English stop words.
# Negation words are deliberately preserved because they can change
# the sentiment polarity of a review.
STOP_WORDS = set(stopwords.words("english")) - {
    "no",
    "nor",
    "not",
    "never",
}

LEMMATIZER = WordNetLemmatizer()


def clean_text_enhanced(text: str) -> str:
    """
    Apply enhanced NLP preprocessing to a movie review.

    The preprocessing steps are:
    1. Remove HTML tags.
    2. Remove URLs.
    3. Convert text to lowercase.
    4. Keep alphabetic tokens and selected negation words.
    5. Remove stop words while preserving important negation terms.
    6. Lemmatize remaining tokens.
    7. Normalize whitespace.

    Parameters
    ----------
    text : str
        Raw movie review text.

    Returns
    -------
    str
        Enhanced preprocessed review text.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Convert to lowercase.
    text = text.lower()

    # Extract alphabetic tokens.
    tokens = re.findall(r"[a-z]+", text)

    # Remove stop words while preserving negation.
    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

    # Lemmatize the remaining tokens.
    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
    ]

    return " ".join(tokens)


def preprocess_reviews_enhanced(texts):
    """
    Apply enhanced preprocessing to a collection of reviews.

    Parameters
    ----------
    texts : iterable
        Collection of raw movie-review texts.

    Returns
    -------
    list[str]
        Enhanced preprocessed review texts.
    """
    return [
        clean_text_enhanced(text)
        for text in texts
    ]


if __name__ == "__main__":
    sample = (
        "<br /><br />This movie was NOT boring! "
        "The actors were running beautifully. "
        "Visit https://example.com for information."
    )

    cleaned = clean_text_enhanced(sample)

    print("Original:")
    print(sample)

    print("\nEnhanced preprocessing:")
    print(cleaned)

    print("\nNegation preservation check:")
    print("'not' preserved:", "not" in cleaned)