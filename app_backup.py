import streamlit as st
import pickle
import re
import sqlite3
import cv2
import numpy as np
import pandas as pd

from nltk.corpus import stopwords

# Page Config
st.set_page_config(
    page_title="AI Fake News & Deepfake Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load Models
model = pickle.load(open("models/model.pkl", "rb"))

deepfake_model = pickle.load(
    open("models/deepfake_model.pkl", "rb")
)

vectorizer = pickle.load(
    open("models/vectorizer.pkl", "rb")
)

stop_words = set(stopwords.words("english"))

# Text Cleaning Function
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    return " ".join(filtered_words)


# Title
st.title("🛡️ AI Fake News & Deepfake Detection System")

st.markdown("""
Detect fake news articles using NLP and identify manipulated images using Machine Learning.
""")

# Tabs
tab1, tab2 = st.tabs(
    ["📰 Fake News Detection", "🖼️ Deepfake Detection"]
)

# ==========================
# FAKE NEWS TAB
# ==========================
with tab1:

    st.subheader("📰 Fake News Detection")

    news = st.text_area("Enter News Article")

    original_news = news

    if st.button("Predict News"):

        news = clean_text(news)

        news = vectorizer.transform([news])

        prediction = model.predict(news)

        probability = model.predict_proba(news)

        confidence = max(probability[0]) * 100

        if prediction[0] == 0:
            st.error("Prediction : FAKE NEWS")
            result = "FAKE NEWS"
        else:
            st.success("Prediction : TRUE NEWS")
            result = "TRUE NEWS"

        st.metric(
            "Confidence Score",
            f"{round(confidence,2)}%"
        )

        # Save Prediction
        conn = sqlite3.connect("news.db")

        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO prediction_history(news,prediction,confidence) VALUES(?,?,?)",
            (
                original_news,
                result,
                round(confidence, 2)
            )
        )

        conn.commit()

        conn.close()

        st.info(
            """
This prediction is generated using a Machine Learning model.

The model is trained on historical labeled news data.

Very recent or unseen news articles may not always be classified correctly.

Final verification should always be done using trusted news sources.
            """
        )

    st.write("## Prediction History")

    if st.button("View Prediction History"):

        conn = sqlite3.connect("news.db")

        history = conn.execute(
            "SELECT news, prediction, confidence FROM prediction_history"
        ).fetchall()

        conn.close()

        df = pd.DataFrame(
            history,
            columns=["News", "Prediction", "Confidence"]
        )

        st.dataframe(df)

# ==========================
# DEEPFAKE TAB
# ==========================
with tab2:

    st.subheader("🖼️ Deepfake Image Detection")

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            width=350
        )

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        img = cv2.resize(
            img,
            (128, 128)
        )

        img = img.reshape(1, -1)

        prediction = deepfake_model.predict(img)

        if prediction[0] == 0:
            st.success("REAL IMAGE")
        else:
            st.error("FAKE IMAGE")