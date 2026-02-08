# Emotion-to-Emoji Model

Small demo project: train a simple text classifier that predicts the emotion of a short sentence and maps it to an emoji.

What’s included
- `model.py` — training, saving, loading, and prediction helpers
- `train.py` — script with a tiny dataset to train and save the model
- `predict.py` — convenience function `predict_emotion(text)` returning (label, emoji)
- `emoji_map.json` — mapping from emotion labels to emoji
- `tests/test_predict.py` — pytest tests (happy path)
- `requirements.txt` — python dependencies

Quick start

1. (Optional) Create a virtualenv and activate it.
2. Install deps: pip install -r requirements.txt
3. Train: python train.py
4. Predict: python -c "from predict import predict_emotion; print(predict_emotion('I am so happy'))"

Notes
- This is a small demo intended for quick experimentation.
