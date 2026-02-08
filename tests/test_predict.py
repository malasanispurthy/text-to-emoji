import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict import predict_emotion


def test_predict_training_sentences():
    # Sentences similar to training examples should map to expected labels
    cases = [
        ("I am so happy and excited", "joy"),
        ("This makes me very sad", "sadness"),
        ("I am furious about this", "anger"),
        ("I'm scared and nervous", "fear"),
        ("What a wonderful surprise", "surprise"),
        ("I love this a lot", "love"),
    ]
    for text, expected in cases:
        label, emoji = predict_emotion(text)
        assert label == expected, f"{text!r} -> {label} expected {expected}"
