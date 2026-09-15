from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Load trained chatbot files
# =========================================================

model = joblib.load("chatbot_model.pkl")
vectorizer = joblib.load("chatbot_vectorizer.pkl")
response_data = joblib.load("chatbot_responses.pkl")


# =========================================================
# Clean user message
# =========================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# Clean chatbot response
# =========================================================

def clean_response(response):

    response = str(response)

    # -----------------------------------------------------
    # Known dataset placeholders
    # -----------------------------------------------------

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

    # Replace known placeholders
    for old, new in replacements.items():
        response = response.replace(old, new)

    # -----------------------------------------------------
    # Remove any unknown {{...}} placeholders
    # -----------------------------------------------------

    response = re.sub(
        r"\{\{[^}]+\}\}",
        "the requested information",
        response
    )

    # -----------------------------------------------------
    # Generic duplicate-word removal
    #
    # Example:
    # "your your order number"
    # becomes:
    # "your order number"
    # -----------------------------------------------------

    response = re.sub(
        r"\b(\w+)(\s+\1\b)+",
        r"\1",
        response,
        flags=re.IGNORECASE
    )

    # -----------------------------------------------------
    # Generic repeated-phrase removal
    #
    # Example:
    # "order status order status"
    # becomes:
    # "order status"
    #
    # Checks repeated phrases from 5 words down to 2 words.
    # -----------------------------------------------------

    words = response.split()

    cleaned_words = []
    i = 0

    while i < len(words):

        removed = False

        for phrase_length in range(5, 1, -1):

            if i + phrase_length * 2 <= len(words):

                first_phrase = [
                    re.sub(r"[^\w]", "", word.lower())
                    for word in words[i:i + phrase_length]
                ]

                second_phrase = [
                    re.sub(r"[^\w]", "", word.lower())
                    for word in words[
                        i + phrase_length:
                        i + phrase_length * 2
                    ]
                ]

                if first_phrase == second_phrase:

                    cleaned_words.extend(
                        words[i:i + phrase_length]
                    )

                    i += phrase_length * 2
                    removed = True
                    break

        if not removed:

            cleaned_words.append(words[i])
            i += 1

    response = " ".join(cleaned_words)

    # -----------------------------------------------------
    # Generic cleanup for placeholder replacement
    #
    # Example:
    # "order number your order number"
    # can become:
    # "your order number"
    # -----------------------------------------------------

    placeholder_phrases = [
        "your order number",
        "your tracking number",
        "order status",
        "customer support hours",
        "customer support phone number",
        "our online account portal",
        "order section",
        "our website"
    ]

    for phrase in placeholder_phrases:

        # Remove duplicate occurrence when the same phrase
        # appears twice with text such as:
        # "order number your order number"

        pattern = (
            r"\b(?:"
            r"order number\s+"
            r")?"
            + re.escape(phrase)
            + r"\s+"
            + re.escape(phrase)
            + r"\b"
        )

        response = re.sub(
            pattern,
            phrase,
            response,
            flags=re.IGNORECASE
        )

    # -----------------------------------------------------
    # Normalize spaces
    # -----------------------------------------------------

    response = re.sub(r"\s+", " ", response)

    # Remove spaces before punctuation
    response = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        response
    )

    # Fix multiple punctuation
    response = re.sub(
        r"([.!?]){2,}",
        r"\1",
        response
    )

    return response.strip()


# =========================================================
# Create response search system
# =========================================================

response_vectorizers = {}
response_matrices = {}


for intent, responses in response_data.items():

    response_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    response_matrix = response_vectorizer.fit_transform(
        responses
    )

    response_vectorizers[intent] = response_vectorizer
    response_matrices[intent] = response_matrix


# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(
    title="Customer Support Chatbot API",
    description="ML based Customer Support Chatbot",
    version="2.2"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Request Model
# =========================================================

class ChatRequest(BaseModel):
    message: str


# =========================================================
# Home
# =========================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "Customer Support Chatbot API is running"
    }


# =========================================================
# Prediction
# =========================================================

@app.post("/predict")
def predict(request: ChatRequest):

    # -----------------------------------------------------
    # Clean user message
    # -----------------------------------------------------

    message = clean_text(request.message)

    # -----------------------------------------------------
    # Convert message into TF-IDF
    # -----------------------------------------------------

    message_tfidf = vectorizer.transform([message])

    # -----------------------------------------------------
    # Predict intent
    # -----------------------------------------------------

    predicted_intent = model.predict(message_tfidf)[0]

    # -----------------------------------------------------
    # Get responses for predicted intent
    # -----------------------------------------------------

    responses = response_data[predicted_intent]

    # -----------------------------------------------------
    # Get response vectorizer and matrix
    # -----------------------------------------------------

    response_vectorizer = response_vectorizers[predicted_intent]

    response_matrix = response_matrices[predicted_intent]

    # -----------------------------------------------------
    # Convert user message for response similarity
    # -----------------------------------------------------

    user_vector = response_vectorizer.transform(
        [message]
    )

    # -----------------------------------------------------
    # Calculate cosine similarity
    # -----------------------------------------------------

    similarities = cosine_similarity(
        user_vector,
        response_matrix
    )[0]

    # -----------------------------------------------------
    # Find most relevant response
    # -----------------------------------------------------

    best_index = similarities.argmax()

    selected_response = responses[best_index]

    # -----------------------------------------------------
    # Clean selected response
    # -----------------------------------------------------

    selected_response = clean_response(
        selected_response
    )

    # -----------------------------------------------------
    # Similarity score
    # -----------------------------------------------------

    similarity_score = float(
        similarities[best_index]
    )

    # -----------------------------------------------------
    # Return API response
    # -----------------------------------------------------

    return {
        "message": request.message,
        "intent": predicted_intent,
        "response": selected_response,
        "similarity_score": round(similarity_score, 4)
    }
