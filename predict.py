"""Load model and predict emotion, mapping to emoji."""
from pathlib import Path
import json
from typing import Tuple

from model import load_model, train_model, save_model, predict as model_predict

ROOT = Path(__file__).parent
EMOJI_MAP_FILE = ROOT / "emoji_map.json"
MODEL_FILE = ROOT / "artifacts" / "model.joblib"


def _load_emoji_map():
    with open(EMOJI_MAP_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict_emotion(text: str) -> Tuple[str, str]:
    """Return (label, emoji) for the given text.

    If no trained model artifact exists, train a small demo model on-the-fly.
    """
    pipeline = load_model()
    if pipeline is None:
        # lightweight on-the-fly training using the same tiny dataset as train.py
        demo = [
            ("I am so happy and excited", "joy"),
            ("This makes me very sad", "sadness"),
            ("I am furious about this", "anger"),
            ("I'm scared and nervous", "fear"),
            ("What a wonderful surprise", "surprise"),
            ("I love this a lot", "love"),
            ("It's okay, not great", "neutral"),
        ]
        pipeline = train_model(demo)
        save_model(pipeline)

    label = model_predict(pipeline, text)
    emoji_map = _load_emoji_map()
    emoji = emoji_map.get(label, "")
    return label, emoji


if __name__ == '__main__':
    examples = [
        "I am so happy",
        "I feel very sad today",
        "I'm absolutely furious",
        "This is surprising",
    ]
    for ex in examples:
        label, emoji = predict_emotion(ex)
        print(f"{ex!r} -> {label} {emoji}")
