"""Train and evaluate a lightweight, explainable text-classification baseline."""
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA = Path(__file__).resolve().parents[1] / "data" / "tickets.csv"


class ModelBundle(NamedTuple):
    pipeline: Pipeline
    categories: tuple[str, ...]


def load_examples() -> pd.DataFrame:
    frame = pd.read_csv(DATA)
    expected = {"text", "category"}
    if not expected.issubset(frame.columns):
        raise ValueError(f"Expected dataset columns: {sorted(expected)}")
    frame = frame.dropna(subset=["text", "category"])
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame.loc[frame["text"].ne("")]
    return frame


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])


@lru_cache(maxsize=1)
def get_model() -> ModelBundle:
    examples = load_examples()
    model = build_pipeline()
    model.fit(examples["text"], examples["category"])
    return ModelBundle(model, tuple(str(c) for c in model.classes_))


def classify_text(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("The message cannot be blank")
    bundle = get_model()
    probs = bundle.pipeline.predict_proba([text])[0]
    best = int(probs.argmax())
    return {
        "category": bundle.categories[best],
        "score": round(float(probs[best]), 4),
        "alternatives": [
            {"category": name, "score": round(float(prob), 4)}
            for name, prob in sorted(zip(bundle.categories, probs), key=lambda x: -x[1])
        ],
    }


def evaluate() -> dict:
    examples = load_examples()
    x_train, x_test, y_train, y_test = train_test_split(
        examples["text"], examples["category"],
        test_size=0.25, random_state=42, stratify=examples["category"]
    )
    model = build_pipeline().fit(x_train, y_train)
    predicted = model.predict(x_test)
    return {
        "test_size": len(x_test),
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "classification_report": classification_report(y_test, predicted, zero_division=0),
    }
