"""Extract brands/places (named entities) and product features from reviews with spaCy,
and attach the sentiment predicted by the fine-tuned model (if ./bert-sentiment exists).

Setup:   pip install spacy && python -m spacy download en_core_web_sm
Usage:   python ner_features.py        (run train_bert.py first to get sentiment labels)
"""
import os
from collections import Counter

import spacy

REVIEWS = [
    "The battery life on this Samsung phone is amazing, but the camera is disappointing.",
    "Apple support in Hyderabad replaced my charger quickly. Great service!",
    "The screen cracked after a week and the delivery from Amazon was late.",
]

IGNORE_LABELS = {"DATE", "TIME", "CARDINAL"}


def load_sentiment(path="bert-sentiment"):
    """Return a function text -> 'positive'/'negative' (or 'n/a' if no trained model is found)."""
    if not os.path.isdir(path):
        return lambda text: "n/a"

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict(text):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        return "positive" if logits.argmax(-1).item() == 1 else "negative"

    return predict


def analyze(nlp, review):
    """Return (entities, features). Features keep compound nouns ('battery life') and skip dates."""
    doc = nlp(review)
    entities = [(e.text, e.label_) for e in doc.ents if e.label_ not in IGNORE_LABELS]
    skip = {t.i for e in doc.ents if e.label_ in IGNORE_LABELS for t in e}
    features = []
    for chunk in doc.noun_chunks:
        if chunk.root.i in skip or chunk.root.pos_ != "NOUN":
            continue
        features.append(" ".join(t.lemma_.lower() for t in chunk if t.pos_ == "NOUN"))
    return entities, features


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    sentiment = load_sentiment()
    counts = Counter()
    for review in REVIEWS:
        entities, features = analyze(nlp, review)
        counts.update(features)
        print(sentiment(review), "|", entities, "|", features)
    print("\nMost mentioned features:", counts.most_common(5))
