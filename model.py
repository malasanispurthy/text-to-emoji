"""Model helper: train, save, load, predict.

Implements a TF-IDF + LogisticRegression pipeline and helpers to persist artifacts.
"""
from typing import List, Tuple
import json
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


ARTIFACT_PATH = Path(__file__).parent / "artifacts"
ARTIFACT_PATH.mkdir(exist_ok=True)
MODEL_FILE = ARTIFACT_PATH / "model.joblib"


def train_model(data: List[Tuple[str, str]]):
    """Train and return a fitted sklearn pipeline.

    data: list of (text, label)
    """
    texts, labels = zip(*data)
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    clf = LogisticRegression(max_iter=1000)
    pipeline = make_pipeline(vect, clf)
    pipeline.fit(texts, labels)
    return pipeline


def save_model(pipeline):
    joblib.dump(pipeline, MODEL_FILE)


def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


def predict(pipeline, text: str) -> str:
    """Return predicted label for text using supplied pipeline."""
    return pipeline.predict([text])[0]
