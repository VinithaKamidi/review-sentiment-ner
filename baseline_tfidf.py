"""Baseline: TF-IDF + Logistic Regression sentiment classifier.

Uses the same 5,000 train / 1,000 test split as train_bert.py so the two are comparable.
"""
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

TRAIN_SIZE, TEST_SIZE, SEED = 5000, 1000, 42

ds = load_dataset("fancyzhx/amazon_polarity")
train = ds["train"].shuffle(seed=SEED).select(range(TRAIN_SIZE))
test = ds["test"].shuffle(seed=SEED).select(range(TEST_SIZE))

# Turn each review into numbers based on its words and word pairs
vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
X_train = vec.fit_transform(train["content"])
X_test = vec.transform(test["content"])

# Learn which words signal positive (1) or negative (0)
clf = LogisticRegression(max_iter=1000).fit(X_train, train["label"])
pred = clf.predict(X_test)

print("Baseline accuracy:", round(accuracy_score(test["label"], pred), 3))
print("Baseline F1:", round(f1_score(test["label"], pred), 3))
