import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def clean_text_spacy(text: str, remove_stopwords: bool = True) -> str:
    doc = nlp(text.lower())
    tokens = [
        t.lemma_ for t in doc
        if t.is_alpha and (not remove_stopwords or not t.is_stop)
    ]
    return " ".join(tokens)