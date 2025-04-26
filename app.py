import streamlit as st
import numpy as np
import pickle
import emoji
import re
import pandas as pd
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tensorflow.keras.models import load_model
import nltk
from pymongo import MongoClient

# ==================== MongoDB Setup ====================

# Get MongoDB URI from Streamlit secrets
MONGO_URI = st.secrets["mongo"]["uri"]

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)

# Select the database and collection
db = client['admin1234']  # Replace with your actual database name
collection = db['user_feedback']  # Replace with your actual collection name

# ==================== Model and Preprocessing Setup ====================

nltk.download('wordnet')
nltk.download('omw-1.4')

# Load artifacts
model = load_model("model_components/emotion_nn_model.h5")
with open("model_components/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model_components/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Sentiment Mapping
sentiment_mapping = {
    "sad": "Negative",
    "fear": "Negative",
    "anger": "Negative",
    "love": "Positive",
    "joy": "Positive",
    "surprise": "Neutral"
}

negation_words = {"not", "no", "never", "none", "nothing", "nobody", "neither", "nowhere", "without"}

def get_antonym(word):
    antonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name().lower())
    return next(iter(antonyms), None)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = emoji.demojize(text)
    text = text.replace(":", " ").lower()
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()
    processed_words = []
    negate = False

    for word in words:
        if word in negation_words:
            negate = True
            continue
        if negate:
            antonym = get_antonym(word)
            if antonym and antonym not in ENGLISH_STOP_WORDS:
                processed_words.append(antonym)
            else:
                processed_words.append(f"neg_{word}")
            negate = False
        else:
            if word not in ENGLISH_STOP_WORDS:
                processed_words.append(word)

    return ' '.join(processed_words)

def predict_emotion(text):
    clean_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([clean_text]).toarray()
    prediction = model.predict(vectorized_text)[0]
    predicted_index = np.argmax(prediction)
    predicted_emotion = label_encoder.classes_[predicted_index]
    predicted_sentiment = sentiment_mapping.get(predicted_emotion.lower(), "Neutral")
    return predicted_emotion.capitalize(), predicted_sentiment, prediction

# ==================== MongoDB Storage for Feedback ====================

def save_feedback(text, predicted, actual):
    feedback_data = {
        "Text": text,
        "Predicted": predicted,
        "Actual": actual
    }

    # Insert feedback into MongoDB collection
    collection.insert_one(feedback_data)

# ==================== Streamlit App ====================

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #2c4d34, #8a4c2b);
        background-attachment: fixed;
    }
    @media (max-width: 640px) {
    .outputs{
         display: flex;align-items: center;   
    }
    .outputs h4{
         padding:0;   
    }
}
    </style>
""", unsafe_allow_html=True)

# App UI
st.markdown("<h1 style='text-align: center;'> Emotion & Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>From the <b>frustrated</b> people having no life </p>", unsafe_allow_html=True)

user_input = st.text_area("Enter a sentence to analyze:", height=150)

if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Hey! Put some nonsense text")
    else:
        emotion, sentiment, probs = predict_emotion(user_input)

        st.session_state.last_user_input = user_input
        st.session_state.last_predicted_emotion = emotion
        st.session_state.feedback_submitted = False

        # Show results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class='outputs' style='padding: 10px; border-radius: 10px; background-color: #f4f4f4;'>
                    <h4 style='color: rgb(54 41 58);'> Emotion:</h4>
                    <h2 style='color: #ff6347;'>{emotion}</h2>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            sentiment_color = "#2ecc71" if sentiment == "Positive" else "#e74c3c" if sentiment == "Negative" else "#f1c40f"
            st.markdown(f"""
                <div class='outputs' style='padding: 10px; border-radius: 10px; background-color: #f4f4f4;'>
                    <h4 style='color: rgb(54 41 58);'> Sentiment:</h4>
                    <h2 style='color: {sentiment_color};'>{sentiment}</h2>
                </div>
            """, unsafe_allow_html=True)

        # Sidebar: Emotion Probabilities
        st.sidebar.subheader("Emotion Probabilities")
        for emo, prob in zip(label_encoder.classes_, probs):
            st.sidebar.write(f"**{emo.capitalize()}**: {prob*100:.2f}%")
            st.sidebar.progress(float(prob))

# ⭐ Feedback section always shows after prediction ⭐
if st.session_state.last_user_input != "":
    st.markdown("---")
    st.subheader("🔁 Was the prediction correct?")

    if not st.session_state.feedback_submitted:
        feedback = st.radio("Select one:", ["Yes", "No"], horizontal=True, key="feedback_radio")

        if feedback == "No":
            correct_emotion = st.selectbox(
                "Select the correct emotion:",
                ["Love", "Fear", "Joy", "Surprise", "Sad", "Anger", "Neutral"],
                key="correction_select"
            )
            if st.button("Submit Feedback (Correction)"):
                save_feedback(st.session_state.last_user_input, st.session_state.last_predicted_emotion, correct_emotion)
                st.session_state.feedback_submitted = True
                st.success("✅ Thanks for the feedback! We'll use this to improve the model.")
        else:
            if st.button("Submit Feedback (Confirmed)"):

                # save_feedback(st.session_state.last_user_input, st.session_state.last_predicted_emotion, st.session_state.last_predicted_emotion)
                st.session_state.feedback_submitted = True
                st.success("✅ Awesome! Thanks for confirming the prediction.")
    else:
        st.info("✅ Feedback already submitted for this prediction.")
