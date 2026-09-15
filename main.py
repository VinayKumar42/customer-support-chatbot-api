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
    text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# Clean chatbot response
# =========================================================

def clean_response(response):

    response = str(response)

    # =====================================================
    # Placeholder mapping
    # =====================================================

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

    # =====================================================
    # Smart placeholder replacement
    # =====================================================

    for placeholder, replacement in replacements.items():

        if placeholder not in response:
            continue

        # -------------------------------------------------
        # Get placeholder name
        # Example:
        # {{Order Number}} -> Order Number
        # -------------------------------------------------

        placeholder_name = (
            placeholder
            .replace("{{", "")
            .replace("}}", "")
            .strip()
        )

        placeholder_words = placeholder_name.split()

        # -------------------------------------------------
        # Create regex for words immediately before
        # placeholder.
        #
        # Example:
        #
        # "the order number {{Order Number}}"
        #
        # becomes:
        #
        # "the your order number"
        #
        # and then generic cleanup removes unnecessary
        # duplicated context.
        # -------------------------------------------------

        if placeholder_words:

            prefix_pattern = r"\s+".join(
                re.escape(word)
                for word in placeholder_words
            )

            pattern = (
                r"\b"
                + prefix_pattern
                + r"\s*"
                + re.escape(placeholder)
            )

            response = re.sub(
                pattern,
                replacement,
                response,
                flags=re.IGNORECASE
            )

        # -------------------------------------------------
        # Replace any remaining occurrence
        # -------------------------------------------------

        response = response.replace(
            placeholder,
            replacement
        )

    # =====================================================
    # Remove unknown placeholders
    # =====================================================

    response = re.sub(
        r"\{\{[^}]+\}\}",
        "the requested information",
        response
    )

    # =====================================================
    # Generic duplicate word cleanup
    #
    # Examples:
    #
    # your your order
    # order order
    # the the order
    #
    # becomes:
    #
    # your order
    # order
    # the order
    # =====================================================

    response = re.sub(
        r"\b(\w+)(\s+\1\b)+",
        r"\1",
        response,
        flags=re.IGNORECASE
    )

    # =====================================================
    # Generic repeated phrase cleanup
    #
    # Example:
    #
    # "order status order status"
    #
    # becomes:
    #
    # "order status"
    #
    # Checks phrases from 6 words down to 2 words.
    # =====================================================

    words = response.split()

    cleaned_words = []
    i = 0

    while i < len(words):

        removed = False

        for phrase_length in range(6, 1, -1):

            if i + (phrase_length * 2) <= len(words):

                first_phrase = [
                    re.sub(
                        r"[^\w]",
                        "",
                        word.lower()
                    )
                    for word in words[
                        i:i + phrase_length
                    ]
                ]

                second_phrase = [
                    re.sub(
                        r"[^\w]",
                        "",
                        word.lower()
                    )
                    for word in words[
                        i + phrase_length:
                        i + (phrase_length * 2)
                    ]
                ]

                if first_phrase == second_phrase:

                    cleaned_words.extend(
                        words[
                            i:i + phrase_length
                        ]
                    )

                    i += phrase_length * 2
                    removed = True
                    break

        if not removed:

            cleaned_words.append(words[i])
            i += 1

    response = " ".join(cleaned_words)

    # =====================================================
    # Smart semantic-style duplicate cleanup
    #
    # Handles cases like:
    #
    # "order number your order number"
    # "tracking number your tracking number"
    #
    # without requiring a separate replacement for every
    # possible sentence.
    # =====================================================

    common_generated_phrases = [
        "your order number",
        "your tracking number",
        "your account",
        "your email address",
        "your shipping address",
        "your order status",
        "our website",
        "our online account portal",
        "customer support hours",
        "customer support phone number",
        "order section",
        "order status",
    ]

    for phrase in common_generated_phrases:

        escaped_phrase = re.escape(phrase)

        phrase_words = phrase.split()

        if len(phrase_words) >= 2:

            # -------------------------------------------------
            # Detect:
            #
            # "order number your order number"
            #
            # "tracking number your tracking number"
            #
            # "status your order status"
            #
            # The words before the generated phrase are
            # removed when they duplicate part of the phrase.
            # -------------------------------------------------

            first_part = r"\s+".join(
                re.escape(word)
                for word in phrase_words[-2:]
            )

            pattern = (
                r"\b"
                + first_part
                + r"\s+"
                + escaped_phrase
                + r"\b"
            )

            response = re.sub(
                pattern,
                phrase,
                response,
                flags=re.IGNORECASE
            )

        # -------------------------------------------------
        # Detect exact duplicate generated phrase
        #
        # Example:
        #
        # "your order number your order number"
        # -------------------------------------------------

        duplicate_pattern = (
            r"\b"
            + escaped_phrase
            + r"\s+"
            + escaped_phrase
            + r"\b"
        )

        response = re.sub(
            duplicate_pattern,
            phrase,
            response,
            flags=re.IGNORECASE
        )

    # =====================================================
    # Generic cleanup for common duplicated structure
    #
    # Example:
    #
    # "the order number your order number"
    #
    # -> "your order number"
    # =====================================================

    response = re.sub(
        r"\b(?:the\s+)?order\s+number\s+your\s+order\s+number\b",
        "your order number",
        response,
        flags=re.IGNORECASE
    )

    response = re.sub(
        r"\b(?:the\s+)?tracking\s+number\s+your\s+tracking\s+number\b",
        "your tracking number",
        response,
        flags=re.IGNORECASE
    )

    # =====================================================
    # Remove extra spaces
    # =====================================================

    response = re.sub(
        r"\s+",
        " ",
        response
    )

    # =====================================================
    # Remove spaces before punctuation
    # =====================================================

    response = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        response
    )

    # =====================================================
    # Remove repeated punctuation
    # =====================================================

    response = re.sub(
        r"([.!?]){2,}",
        r"\1",
        response
    )

    # =====================================================
    # Final cleanup
    # =====================================================

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
    version="2.3"
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

    message_tfidf = vectorizer.transform(
        [message]
    )

    # -----------------------------------------------------
    # Predict intent
    # -----------------------------------------------------

    predicted_intent = model.predict(
        message_tfidf
    )[0]

    # -----------------------------------------------------
    # Get responses for predicted intent
    # -----------------------------------------------------

    responses = response_data[
        predicted_intent
    ]

    # -----------------------------------------------------
    # Get response vectorizer
    # -----------------------------------------------------

    response_vectorizer = (
        response_vectorizers[predicted_intent]
    )

    # -----------------------------------------------------
    # Get response matrix
    # -----------------------------------------------------

    response_matrix = (
        response_matrices[predicted_intent]
    )

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

    selected_response = responses[
        best_index
    ]

    # -----------------------------------------------------
    # Clean response
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
    # Return result
    # -----------------------------------------------------

    return {
        "message": request.message,
        "intent": predicted_intent,
        "response": selected_response,
        "similarity_score": round(
            similarity_score,
            4
        )
    }
