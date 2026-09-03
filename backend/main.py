# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from transformers import pipeline


app = FastAPI()


# Allow our Next.js frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the Hugging Face sentiment model
classifier = pipeline("sentiment-analysis")


class SentimentRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Sentiment API is running"}


@app.post("/predict")
def predict(request: SentimentRequest):
    result = classifier(request.text)

    return {
        "label": result[0]["label"],
        "score": result[0]["score"],
    }