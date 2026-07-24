import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def get_all_query(title, author, text):

    title = "" if title is None else str(title).strip()
    author = "" if author is None else str(author).strip()
    text = "" if text is None else str(text).strip()

    total = f"{title} {author} {text}"

    return [total]


def remove_punctuation_stopwords_lemma(sentence):

    sentence = str(sentence).lower()

    if not sentence.strip():
        return ""

    sentence = re.sub(r"[^\w\s]", "", sentence)

    words = nltk.word_tokenize(sentence)

    words = [
        word for word in words
        if word not in stop_words
    ]

    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(cleaned_words)