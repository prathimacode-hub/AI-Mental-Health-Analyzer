"""
AI Mental Health Support Analyzer - Production Streamlit Application

Author: Prathima
Project: AI Mental Health Streamlit App

Features:
- NLP Mental Health Prediction
- Emotion Mapping
- Risk & Severity Detection
- Emergency Alert
- Helpline Support
- Confidence Interpretation
- Dashboard
- History Tracking
- Download Report
- Clean UI
- Error Handling
- Production Ready
"""

# ============================================
# IMPORT LIBRARIES
# ============================================

# ============================================
# IMPORTS
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

from src.utils import (
    suggest_help,
    map_emotion,
    calculate_risk,
    detect_emergency,
    get_helpline,
    interpret_confidence
)

from src.config import (
    MODEL_PATH,
    TFIDF_PATH,
    ENCODER_PATH,
    APP_TITLE,
    APP_DESCRIPTION
)

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide"
)

# ============================================
# HISTORY PATH
# ============================================

HISTORY_PATH = "models/history.csv"

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_resources():

    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        tfidf = pickle.load(open(TFIDF_PATH, "rb"))
        encoder = pickle.load(open(ENCODER_PATH, "rb"))

        return model, tfidf, encoder

    except Exception as e:
        st.error("❌ Model files not found. Please run training first.")
        st.stop()


model, tfidf, encoder = load_resources()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🧠 Mental Health AI")

st.sidebar.markdown("---")

st.sidebar.subheader("About")

st.sidebar.info(APP_DESCRIPTION)

st.sidebar.markdown("---")

st.sidebar.subheader("Disclaimer")

st.sidebar.warning(
    "This tool is for educational purposes only and not medical advice."
)

# ============================================
# MAIN TITLE
# ============================================

st.title(APP_TITLE)
st.markdown("### Understand emotional well-being using AI & NLP")
st.markdown("---")

# ============================================
# TABS
# ============================================

tab1, tab2, tab3 = st.tabs(
    ["📝 Analyze", "📊 Dashboard", "📘 About"]
)

# ============================================
# TAB 1
# ============================================

with tab1:

    st.subheader("Enter your thoughts")

    user_input = st.text_area(
        "How are you feeling today?",
        height=150,
        placeholder="Example: I feel stressed and anxious about my work"
    )

    col1, col2 = st.columns(2)

    analyze = col1.button("🔍 Analyze")
    clear = col2.button("🧹 Clear")

    if clear:
        st.rerun()

    if analyze:

        if user_input.strip() == "":
            st.warning("Please enter text")

        else:

            try:

                # ====================================
                # TEXT TRANSFORM
                # ====================================

                text_vector = tfidf.transform([user_input])

                prediction_encoded = model.predict(text_vector)[0]

                label = encoder.inverse_transform(
                    [prediction_encoded]
                )[0]

                # ====================================
                # CONFIDENCE
                # ====================================

                try:
                    prob = model.predict_proba(text_vector)
                    confidence = float(np.max(prob) * 100)
                except:
                    confidence = 80.0

                confidence_text = interpret_confidence(confidence)

                # ====================================
                # EMOTION
                # ====================================

                emotion = map_emotion(label)

                risk, severity = calculate_risk(label, confidence)

                suggestions = suggest_help(label)

                helpline = get_helpline()

                emergency = detect_emergency(user_input)

                # ====================================
                # RESULT
                # ====================================

                st.markdown("---")
                st.subheader("Prediction Result")

                if label.lower() == "depression":
                    st.error("🔴 Depression Detected")

                elif label.lower() == "stress":
                    st.warning("🟠 Stress Detected")

                elif label.lower() == "anxiety":
                    st.warning("🟡 Anxiety Detected")

                else:
                    st.success("🟢 Normal Emotional State")

                # ====================================
                # METRICS
                # ====================================

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Emotion", emotion["emotion"])
                col2.metric("Risk Level", risk)
                col3.metric("Severity", severity)
                col4.metric("Confidence", f"{confidence:.2f}%")

                st.write("Confidence:", confidence_text)

                st.markdown("---")

                # ====================================
                # EMERGENCY
                # ====================================

                if emergency:
                    st.error(
                        "🚨 Emergency detected. Seek immediate help."
                    )

                # ====================================
                # SUGGESTIONS
                # ====================================

                st.subheader("Suggested Support")

                for s in suggestions:
                    st.write("✔", s)

                st.markdown("---")

                # ====================================
                # HELPLINES
                # ====================================

                st.subheader("Helpline Support")

                for h in helpline["helplines"]:

                    st.info(
                        f"{h['name']} : {h['number']}"
                    )

                # ====================================
                # SAVE HISTORY
                # ====================================

                new_data = pd.DataFrame({

                    "text": [user_input],
                    "prediction": [label],
                    "confidence": [confidence],
                    "risk": [risk],
                    "severity": [severity],
                    "time": [datetime.now()]
                })

                if os.path.exists(HISTORY_PATH):

                    old = pd.read_csv(HISTORY_PATH)
                    updated = pd.concat([old, new_data])

                else:
                    updated = new_data

                updated.to_csv(HISTORY_PATH, index=False)

                st.success("Saved to dashboard")

            except Exception as e:

                st.error("Prediction failed")
                st.write(e)

# ============================================
# TAB 2
# ============================================

with tab2:

    st.subheader("Prediction Dashboard")

    if os.path.exists(HISTORY_PATH):

        history = pd.read_csv(HISTORY_PATH)

        st.write("Total Predictions:", len(history))

        st.dataframe(history)

        st.markdown("---")

        st.subheader("Label Distribution")

        st.bar_chart(
            history["prediction"].value_counts()
        )

        st.markdown("---")

        st.subheader("Confidence Trend")

        st.line_chart(history["confidence"])

        csv = history.to_csv(index=False).encode()

        st.download_button(
            "Download Report",
            csv,
            "mental_health_report.csv",
            "text/csv"
        )

    else:

        st.info("No history yet")

# ============================================
# TAB 3
# ============================================

with tab3:

    st.subheader("Project Overview")

    st.write(
        """
AI Mental Health Support Analyzer predicts emotional state
from user text using NLP and Machine Learning.

Workflow:

Dataset → TF-IDF → ML Model → Streamlit App
"""
    )

    st.subheader("Technologies")

    st.table(pd.DataFrame({

        "Tech": [
            "Python",
            "Streamlit",
            "Machine Learning",
            "Scikit-learn",
            "TF-IDF",
            "Pandas"
        ]
    }))

    st.subheader("Future Enhancements")

    st.write(
        """
Voice Input  
Chatbot  
Mobile App  
Multi-language Support  
Real-time Counseling
Emotion Graph
"""
    )
