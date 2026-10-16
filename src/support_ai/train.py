from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
MODEL_PATH = Path("models/triage.joblib")


def build_model():
    # Chosen by 5-fold CV on train (macro F1 0.877 ± 0.006); see 01_explore.ipynb
    return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000, C=10))


def main():
    train = pd.read_csv(f"{BASE}/train.csv")
    test = pd.read_csv(f"{BASE}/test.csv")

    model = build_model()
    model.fit(train["text"], train["category"])

    preds = model.predict(test["text"])
    print("test accuracy:", round(accuracy_score(test["category"], preds), 3))
    print("test macro F1:", round(f1_score(test["category"], preds, average="macro"), 3))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("saved to", MODEL_PATH)


if __name__ == "__main__":
    main()