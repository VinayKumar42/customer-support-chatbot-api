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
    # 1. Remove / replace dataset placeholders
    # =====================================================

    replacements = {
        "{{Order Number}}": "your order number",
        "{{Tracking Number}}": "your tracking number",
        "{{Online Company Portal Info}}": "your online account",
        "{{Online Order Interaction}}": "your order",
        "{{Order Status}}": "your order status",
        "{{Customer Support Hours}}": "our support hours",
        "{{Customer Support Phone Number}}": "our support phone number",
        "{{Website URL}}": "our website",
    }

    # -----------------------------------------------------
    # First handle phrases where placeholder is duplicated
    # with the same information immediately before it.
    #
    # Example:
    # "with the order number {{Order Number}}"
    #
    # becomes:
    # "with your order number"
    # -----------------------------------------------------

    response = re.sub(
        r"\b(?:the\s+)?order\s+number\s*\{\{Order Number\}\}",
        "your order number",
        response,
        flags=re.IGNORECASE
    )

    response = re.sub(
        r"\b(?:the\s+)?tracking\s+number\s*\{\{Tracking Number\}\}",
        "your tracking number",
        response,
        flags=re.IGNORECASE
    )

    response = re.sub(
        r"\b(?:your\s+)?online\s+account\s*\{\{Online Company Portal Info\}\}",
        "your online account",
        response,
        flags=re.IGNORECASE
    )

    # -----------------------------------------------------
    # Replace remaining placeholders
    # -----------------------------------------------------

    for placeholder, replacement in replacements.items():

        response = response.replace(
            placeholder,
            replacement
        )

        response = response.replace(
            placeholder.lower(),
            replacement
        )

    # =====================================================
    # 2. Remove unknown placeholders
    # =====================================================

    response = re.sub(
        r"\{\{[^{}]+\}\}",
        "",
        response
    )

    # =====================================================
    # 3. Fix broken text caused by placeholder removal
    # =====================================================

    broken_patterns = [

        # -----------------------------------------------
        # Order number related
        # -----------------------------------------------

        (
            r"\bwith\s+(?:the\s+)?order\s+number\s+your\b",
            "with your"
        ),

        (
            r"\b(?:the\s+)?order\s+number\s+your\b",
            "your"
        ),

        (
            r"\bwith\s+your\s+order\s+number\s+your\b",
            "with your order number"
        ),

        (
            r"\byour\s+order\s+number\s+your\s+order\s+number\b",
            "your order number"
        ),

        # -----------------------------------------------
        # Tracking number
        # -----------------------------------------------

        (
            r"\bwith\s+(?:the\s+)?tracking\s+number\s+your\b",
            "with your"
        ),

        (
            r"\b(?:the\s+)?tracking\s+number\s+your\b",
            "your"
        ),

        # -----------------------------------------------
        # Generic possessive duplication
        # -----------------------------------------------

        (
            r"\bthe\s+(your|my|our|their)\b",
            r"\1"
        ),

        (
            r"\ba\s+(your|my|our|their)\b",
            r"\1"
        ),

        (
            r"\ban\s+(your|my|our|their)\b",
            r"\1"
        ),
    ]

    for pattern, replacement in broken_patterns:

        response = re.sub(
            pattern,
            replacement,
            response,
            flags=re.IGNORECASE
        )

    # =====================================================
    # 4. Remove duplicate consecutive words
    # =====================================================

    words = response.split()

    cleaned_words = []

    for word in words:

        normalized_word = re.sub(
            r"[^\w]",
            "",
            word.lower()
        )

        if cleaned_words:

            previous_normalized = re.sub(
                r"[^\w]",
                "",
                cleaned_words[-1].lower()
            )

            if (
                normalized_word
                and normalized_word == previous_normalized
            ):
                continue

        cleaned_words.append(word)

    response = " ".join(cleaned_words)

    # =====================================================
    # 5. Remove duplicate multi-word phrases
    #
    # Generic duplicate detection.
    #
    # Example:
    #
    # "your order number your order number"
    #
    # becomes:
    #
    # "your order number"
    # =====================================================

    for phrase_length in range(8, 1, -1):

        words = response.split()

        cleaned_words = []
        i = 0

        while i < len(words):

            if i + (phrase_length * 2) <= len(words):

                first = words[
                    i:i + phrase_length
                ]

                second = words[
                    i + phrase_length:
                    i + (phrase_length * 2)
                ]

                first_normalized = [
                    re.sub(
                        r"[^\w]",
                        "",
                        x.lower()
                    )
                    for x in first
                ]

                second_normalized = [
                    re.sub(
                        r"[^\w]",
                        "",
                        x.lower()
                    )
                    for x in second
                ]

                if first_normalized == second_normalized:

                    cleaned_words.extend(first)

                    i += phrase_length * 2

                    continue

            cleaned_words.append(words[i])

            i += 1

        response = " ".join(cleaned_words)

    # =====================================================
    # 6. Remove repeated nearby phrases
    # =====================================================

    words = response.split()

    for phrase_length in range(6, 1, -1):

        i = 0

        while i + phrase_length < len(words):

            current_phrase = [
                re.sub(
                    r"[^\w]",
                    "",
                    word.lower()
                )
                for word in words[
                    i:i + phrase_length
                ]
            ]

            found_duplicate = False

            search_start = i + 1

            search_end = min(
                i + phrase_length + 4,
                len(words) - phrase_length + 1
            )

            for j in range(
                search_start,
                search_end
            ):

                next_phrase = [
                    re.sub(
                        r"[^\w]",
                        "",
                        word.lower()
                    )
                    for word in words[
                        j:j + phrase_length
                    ]
                ]

                if current_phrase == next_phrase:

                    del words[
                        j:j + phrase_length
                    ]

                    found_duplicate = True

                    break

            if not found_duplicate:

                i += 1

    response = " ".join(words)

    # =====================================================
    # 7. Fix common grammatical duplication again
    # =====================================================

    response = re.sub(
        r"\b(?:the|a|an)\s+(your|my|our|their)\b",
        r"\1",
        response,
        flags=re.IGNORECASE
    )

    # =====================================================
    # 8. Fix broken sentence joins
    #
    # Example:
    #
    # "... order. To do that ..."
    #
    # Make sure "To do that" starts a sentence.
    # =====================================================

    response = re.sub(
        r"\s+(To do that|To do so|To check this|For this)\b",
        r". \1",
        response,
        flags=re.IGNORECASE
    )

    # =====================================================
    # 9. Remove accidental words before sentence starts
    #
    # Example:
    #
    # "your To do that"
    #
    # becomes:
    #
    # "your. To do that"
    # =====================================================

    response = re.sub(
        r"\b(your|my|our|their)\s+(To do that|To do so)\b",
        r"\1. \2",
        response,
        flags=re.IGNORECASE
    )

    # =====================================================
    # 10. Normalize spaces
    # =====================================================

    response = re.sub(
        r"\s+",
        " ",
        response
    ).strip()

    # =====================================================
    # 11. Remove spaces before punctuation
    # =====================================================

    response = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        response
    )

    # =====================================================
    # 12. Remove repeated punctuation
    # =====================================================

    response = re.sub(
        r"([.!?]){2,}",
        r"\1",
        response
    )

    # =====================================================
    # 13. Fix punctuation followed by lowercase sentence
    # =====================================================

    response = re.sub(
        r"([.!?])\s+([a-z])",
        lambda m: m.group(1) + " " + m.group(2).upper(),
        response
    )

    # =====================================================
    # 14. Final capitalization
    # =====================================================

    if response:

        response = (
            response[0].upper()
            + response[1:]
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

    response_vectorizers[intent] = (
        response_vectorizer
    )

    response_matrices[intent] = (
        response_matrix
    )


# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(
    title="Customer Support Chatbot API",
    description="ML based Customer Support Chatbot",
    version="3.1"
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

    message = clean_text(
        request.message
    )

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
        response_vectorizers[
            predicted_intent
        ]
    )

    # -----------------------------------------------------
    # Get response matrix
    # -----------------------------------------------------

    response_matrix = (
        response_matrices[
            predicted_intent
        ]
    )

    # -----------------------------------------------------
    # Convert user message for similarity
    # -----------------------------------------------------

    user_vector = (
        response_vectorizer.transform(
            [message]
        )
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
    # Clean selected response
    # -----------------------------------------------------

    selected_response = clean_response(
        selected_response
    )

    # -----------------------------------------------------
    # Similarity score
    # -----------------------------------------------------

    similarity_score = float(
        similarities[
            best_index
        ]
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
