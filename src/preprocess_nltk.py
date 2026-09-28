import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
URL_RE   = re.compile(r"https?://\S+|www\.\S+")
HTML_RE  = re.compile(r"<.*?>")
EMAIL_RE = re.compile(r"\S+@\S+")

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    text = text.lower()
    text = URL_RE.sub("", text)
    text = HTML_RE.sub("", text)
    text = EMAIL_RE.sub("", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)