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
    # 1. Replace dataset placeholders
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

    for placeholder, replacement in replacements.items():

        # -------------------------------------------------
        # Placeholder name
        # -------------------------------------------------

        placeholder_name = (
            placeholder
            .replace("{{", "")
            .replace("}}", "")
            .strip()
        )

        # -------------------------------------------------
        # Convert placeholder name into flexible pattern
        #
        # Example:
        # Order Number
        # -------------------------------------------------

        placeholder_words = placeholder_name.split()

        if placeholder_words:

            prefix_pattern = r"\s+".join(
                re.escape(word)
                for word in placeholder_words
            )

            # -------------------------------------------------
            # Detect:
            #
            # "order number {{Order Number}}"
            #
            # Instead of:
            #
            # "order number your order number"
            #
            # make it:
            #
            # "your order number"
            # -------------------------------------------------

            response = re.sub(
                r"\b"
                + prefix_pattern
                + r"\s*"
                + re.escape(placeholder)
                + r"\b",
                replacement,
                response,
                flags=re.IGNORECASE
            )

        # -------------------------------------------------
        # Replace remaining placeholder
        # -------------------------------------------------

        response = response.replace(
            placeholder,
            replacement
        )

    # =====================================================
    # 2. Remove any unknown {{placeholder}}
    # =====================================================

    response = re.sub(
        r"\{\{[^{}]+\}\}",
        "the requested information",
        response
    )

    # =====================================================
    # 3. Normalize spaces
    # =====================================================

    response = re.sub(
        r"\s+",
        " ",
        response
    ).strip()

    # =====================================================
    # 4. Remove consecutive duplicate words
    #
    # Examples:
    #
    # "your your order"
    # "the the order"
    # "order order"
    #
    # -> clean version
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

            if normalized_word and \
               normalized_word == previous_normalized:
                continue

        cleaned_words.append(word)

    response = " ".join(cleaned_words)

    # =====================================================
    # 5. Remove duplicated multi-word phrases
    #
    # Generic:
    #
    # "order status order status"
    # "your order number your order number"
    # "tracking your tracking"
    #
    # Works for 2-8 word phrases.
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
    # 6. Fix duplicate contextual phrases
    #
    # Generic structure:
    #
    # "the your X"
    # "a your X"
    # "an your X"
    # "the my X"
    # "a my X"
    # "the our X"
    #
    # -> "your X"
    #    "my X"
    #    "our X"
    # =====================================================

    response = re.sub(
        r"\b(?:the|a|an)\s+"
        r"(your|my|our|their)\b",
        r"\1",
        response,
        flags=re.IGNORECASE
    )

    # =====================================================
    # 7. Fix repeated possessive structures
    #
    # Example:
    #
    # "your order number your order"
    #
    # Generic detection where the same meaningful phrase
    # starts again.
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

            next_start = i + 1

            # -------------------------------------------------
            # Search nearby repeated phrase
            # -------------------------------------------------

            found_duplicate = False

            for j in range(
                next_start,
                min(
                    i + phrase_length + 3,
                    len(words) - phrase_length + 1
                )
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

                    # -------------------------------------------------
                    # Keep first occurrence and remove second occurrence
                    # -------------------------------------------------

                    del words[
                        j:j + phrase_length
                    ]

                    found_duplicate = True
                    break

            if not found_duplicate:
                i += 1

    response = " ".join(words)

    # =====================================================
    # 8. Clean common grammatical duplication
    #
    # This is generic and not tied to a particular sentence.
    # =====================================================

    response = re.sub(
        r"\b(the|a|an)\s+(your|my|our|their)\b",
        r"\2",
        response,
        flags=re.IGNORECASE
    )

    # =====================================================
    # 9. Remove extra spaces
    # =====================================================

    response = re.sub(
        r"\s+",
        " ",
        response
    ).strip()

    # =====================================================
    # 10. Remove spaces before punctuation
    # =====================================================

    response = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        response
    )

    # =====================================================
    # 11. Remove repeated punctuation
    # =====================================================

    response = re.sub(
        r"([.!?]){2,}",
        r"\1",
        response
    )

    # =====================================================
    # 12. Final capitalization
    # =====================================================

    if response:
        response = response[0].upper() + response[1:]

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
    version="3.0"
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
    # Convert user message for similarity
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
