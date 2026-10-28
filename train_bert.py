"""Fine-tune DistilBERT for review sentiment (positive / negative).

Run on Google Colab with a free GPU (Runtime > Change runtime type > T4 GPU).
Uses the same 5,000 train / 1,000 test split as baseline_tfidf.py.
Saves the trained model to ./bert-sentiment for ner_features.py.
"""
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL = "distilbert-base-uncased"
TRAIN_SIZE, TEST_SIZE, SEED = 5000, 1000, 42

ds = load_dataset("fancyzhx/amazon_polarity")
train = ds["train"].shuffle(seed=SEED).select(range(TRAIN_SIZE))
test = ds["test"].shuffle(seed=SEED).select(range(TEST_SIZE))

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def tokenize(batch):
    return tokenizer(batch["content"], truncation=True, max_length=128)


train_tok = train.map(tokenize, batched=True)
test_tok = test.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}


args = TrainingArguments(
    output_dir="bert-sentiment",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

result = trainer.evaluate()
print("BERT accuracy:", round(result["eval_accuracy"], 3))
print("BERT F1:", round(result["eval_f1"], 3))

# Error analysis: print the first 8 mistakes so you can look for patterns
preds = trainer.predict(test_tok).predictions.argmax(-1)
true = np.array(test["label"])
for i in np.where(preds != true)[0][:8]:
    print("\nTRUE:", true[i], "PRED:", preds[i], "|", test[int(i)]["content"][:200])

trainer.save_model("bert-sentiment")
tokenizer.save_pretrained("bert-sentiment")
