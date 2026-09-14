
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import random

# Load trained chatbot files
model = joblib.load("chatbot_model.pkl")
vectorizer = joblib.load("chatbot_vectorizer.pkl")
response_data = joblib.load("chatbot_responses.pkl")


# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# FastAPI app
app = FastAPI(
    title="Customer Support Chatbot API",
    description="ML based Customer Support Chatbot",
    version="1.0"
)


# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Customer Support Chatbot API is running"
    }


@app.post("/predict")
def predict(request: ChatRequest):

    message = clean_text(request.message)

    message_tfidf = vectorizer.transform([message])

    predicted_intent = model.predict(message_tfidf)[0]

    possible_responses = response_data[predicted_intent]

    selected_response = random.choice(possible_responses)

    return {
        "message": request.message,
        "intent": predicted_intent,
        "response": selected_response
    }
