from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# Load trained chatbot files
# ==============================

model = joblib.load("chatbot_model.pkl")
vectorizer = joblib.load("chatbot_vectorizer.pkl")
response_data = joblib.load("chatbot_responses.pkl")


# ==============================
# Clean user message
# ==============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==============================
# Clean dataset placeholders
# ==============================

def clean_response(response):

    replacements = {
        "{{Order Number}}": "your order number",
        "{{Tracking Number}}": "your tracking number",
        "{{Online Company Portal Info}}": "our online account portal",
        "{{Online Order Interaction}}": "order section",
        "{{Order Status}}": "order status",
        "{{Customer Support Hours}}": "customer support hours",
        "{{Customer Support Phone Number}}": "customer support phone number",
        "{{Website URL}}": "our website",
    }

    for old, new in replacements.items():
        response = response.replace(old, new)

    # Remove any remaining {{...}} placeholders
    response = re.sub(
        r"\{\{[^}]+\}\}",
        "the requested information",
        response
    )

    # Fix duplicate phrases caused by placeholder replacement
    response = response.replace(
    "your your order number",
    "your order number"
    )
    
    response = response.replace(
        "your your tracking number",
        "your tracking number"
    )
    response = response.replace(
        "the order number your order number",
        "your order number"
    )

    response = response.replace(
        "order number your order number",
        "your order number"
    )

    response = response.replace(
        "the tracking number your tracking number",
        "your tracking number"
    )

    response = response.replace(
        "tracking number your tracking number",
        "your tracking number"
    )

    return response


# ==============================
# Create response search system
# ==============================

response_vectorizers = {}
response_matrices = {}

for intent, responses in response_data.items():

    response_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    response_matrix = response_vectorizer.fit_transform(responses)

    response_vectorizers[intent] = response_vectorizer
    response_matrices[intent] = response_matrix


# ==============================
# FastAPI App
# ==============================

app = FastAPI(
    title="Customer Support Chatbot API",
    description="ML based Customer Support Chatbot",
    version="2.1"
)


# ==============================
# CORS
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# Request Model
# ==============================

class ChatRequest(BaseModel):
    message: str


# ==============================
# Home
# ==============================

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "Customer Support Chatbot API is running"
    }


# ==============================
# Prediction
# ==============================

@app.post("/predict")
def predict(request: ChatRequest):

    # Clean user message
    message = clean_text(request.message)

    # Convert user message into TF-IDF
    message_tfidf = vectorizer.transform([message])

    # Predict intent
    predicted_intent = model.predict(message_tfidf)[0]

    # Get responses for predicted intent
    responses = response_data[predicted_intent]

    # Get response vectorizer and matrix
    response_vectorizer = response_vectorizers[predicted_intent]
    response_matrix = response_matrices[predicted_intent]

    # Convert user message for response similarity
    user_vector = response_vectorizer.transform([message])

    # Calculate similarity
    similarities = cosine_similarity(
        user_vector,
        response_matrix
    )[0]

    # Find most relevant response
    best_index = similarities.argmax()

    selected_response = responses[best_index]

    # Clean placeholders
    selected_response = clean_response(selected_response)

    return {
        "message": request.message,
        "intent": predicted_intent,
        "response": selected_response
    }
