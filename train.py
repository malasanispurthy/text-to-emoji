"""Train a tiny demo model and save artifacts."""
from pathlib import Path
from model import train_model, save_model


DEMO_DATA = [
    ("I am so happy and excited", "joy"),
    ("This makes me very sad", "sadness"),
    ("I am furious about this", "anger"),
    ("I'm scared and nervous", "fear"),
    ("What a wonderful surprise", "surprise"),
    ("I love this a lot", "love"),
    ("It's okay, not great", "neutral"),
    ("I'm feeling pretty happy today", "joy"),
    ("Tears are coming, I'm heartbroken", "sadness"),
]


def main():
    print("Training on tiny demo dataset (for demo only)...")
    pipeline = train_model(DEMO_DATA)
    save_model(pipeline)
    print("Saved model to artifacts/model.joblib")


if __name__ == '__main__':
    main()
