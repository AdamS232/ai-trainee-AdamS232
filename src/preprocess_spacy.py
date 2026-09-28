import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def clean_text_spacy(text: str, remove_stopwords: bool = True) -> str:
    doc = nlp(text.lower())
    tokens = [
        t.lemma_ for t in doc
        if t.is_alpha and (not remove_stopwords or not t.is_stop)
    ]
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "<br />I LOVED this movie!!! See https://example.com or email me@test.com. The actors were running."
    print("Input: ", sample)
    print("Output:", clean_text_spacy(sample))