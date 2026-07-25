import re
import nltk

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize resources only once
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def get_all_query(title, author, text):
    """
    Combine title, author and article into a single string.
    """

    title = "" if title is None else str(title).strip()
    author = "" if author is None else str(author).strip()
    text = "" if text is None else str(text).strip()

    return [f"{title} {author} {text}"]


def remove_punctuation_stopwords_lemma(sentence):
    """
    Clean and preprocess text for prediction.
    """

    sentence = str(sentence).lower().strip()

    if not sentence:
        return ""

    # Remove punctuation
    sentence = re.sub(r"[^\w\s]", "", sentence)

    # Tokenize
    words = nltk.word_tokenize(sentence)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)