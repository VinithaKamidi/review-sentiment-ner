# Product Review Sentiment & Entity Tagger

Classifies Amazon product reviews as positive or negative, comparing a classic TF-IDF baseline with a fine-tuned DistilBERT model, then uses spaCy to pull out brands/places and the product features customers mention.

## Files
- `baseline_tfidf.py`: TF-IDF + Logistic Regression baseline
- `train_bert.py`: fine-tunes DistilBERT, prints metrics and the first 8 mistakes, saves the model
- `ner_features.py`: spaCy entities and product features, plus sentiment from the trained model
- `sentiment_ner_walkthrough.ipynb`: the Colab notebook with all steps and their outputs (if present)

## Data
Amazon product reviews (`fancyzhx/amazon_polarity` on Hugging Face). A fixed random subset is used: **5,000 reviews for training, 1,000 for testing** (seed 42). Labels: 1 = positive, 0 = negative.

## Results
Same data split for both models, measured on the 1,000 test reviews:

| Model | Accuracy | F1 |
|-------|----------|----|
| TF-IDF + Logistic Regression | 0.825 | 0.829 |
| DistilBERT (fine-tuned, 2 epochs) | 0.901 | 0.904 |

DistilBERT improved accuracy by about 7.6 percentage points. With 1,000 test reviews, random variation is roughly 2 points, so the gap is meaningful, but this is a single run on a subset of the data.

Training took about 40 seconds on a free Colab T4 GPU.

## Error analysis
I printed the baseline's and BERT's mistakes and found three patterns:
1. **Contrast with something else.** Negative words describing a *different* or *previous* product ("other garages were made so poorly", "I have been using a terrible hole puncher") fool both models into predicting negative for a positive review. BERT fixed 4 of the baseline's first 8 mistakes but not these.
2. **Late or polite negatives.** Reviews that start positive and turn negative ("Prospective buyers, beware! ... I was certainly impressed when it arrived"), or complain without strong negative words ("I expected a hard hitting original. Instead we get ... filler"). Reviews are also cut at 128 tokens, so a late complaint may never be seen.
3. **Questionable labels.** Some reviews (an off-topic comment about a band, a neutral "heads up") don't clearly match their label. As far as I know, the dataset's labels come from star ratings, so text and label can disagree.

A mixed review such as "The battery life is amazing but the camera is disappointing" gets a single label, which is a built-in limit of review-level sentiment.

## Entity and feature extraction (spaCy)
`ner_features.py` finds named entities and noun-based product features, for example brands (Samsung, Apple, Amazon) and features (battery life, camera, screen, delivery).
Issues found while testing: `en_core_web_sm` labelled the city Hyderabad as an organization, and a first version kept only the last word of compound features ("life" instead of "battery life") and listed dates ("a week") as features. The current script keeps compound nouns and skips dates; entity labels from the small model remain imperfect.

## How to run (Google Colab with a free GPU)
```bash
!git clone https://github.com/VinithaKamidi/review-sentiment-ner.git
%cd review-sentiment-ner
!pip install -q transformers datasets accelerate scikit-learn spacy
!python -m spacy download en_core_web_sm
!python baseline_tfidf.py
!python train_bert.py
!python ner_features.py
```
Set Runtime > Change runtime type > T4 GPU before running `train_bert.py`.

## Limitations
- One run on a 5,000-review subset of one domain; no repeated runs or cross-validation.
- Single sentiment label per review; no aspect-level sentiment (for example, positive for battery, negative for camera).
- Reviews are truncated at 128 tokens.
- Small spaCy model; entity and feature extraction use simple rules.

## Ideas to extend
Aspect-level sentiment, a longer max length, training on more data, a custom product-name entity model, and a small dashboard of the most-complained-about features.
