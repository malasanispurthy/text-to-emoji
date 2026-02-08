"""FastAPI prediction endpoint for emotion-to-emoji model."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predict import predict_emotion


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    emoji: str


app = FastAPI(title="Emotion->Emoji API")

# Mount static files under /static so API routes like /predict are not intercepted
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow CORS so the UI (or other origins) can call the API when needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    label, emoji = predict_emotion(req.text)
    return PredictResponse(label=label, emoji=emoji)
