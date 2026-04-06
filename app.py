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

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

from utils import (
    suggest_help,
    map_emotion,
    calculate_risk,
    detect_emergency,
    get_helpline,
    interpret_confidence,
    format_response
)

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="AI Mental Health Support Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# PATHS
# ============================================

MODEL_PATH = "model/mental_health_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
HISTORY_PATH = "model/history.csv"

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_resources():
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        encoder = pickle.load(open(ENCODER_PATH, "rb"))
        return model, encoder
    except Exception as e:
        st.error("❌ Model or Encoder not found. Please run train.py first.")
        st.stop()

model, encoder = load_resources()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🧠 Mental Health AI")

st.sidebar.markdown("---")

st.sidebar.subheader("About")

st.sidebar.info(
    """
AI-powered NLP system that detects mental health condition
from user text and provides support suggestions.
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("Labels")

st.sidebar.write(
    """
🟢 Normal  
🟡 Anxiety  
🟠 Stress  
🔴 Depression
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("Disclaimer")

st.sidebar.warning(
    """
This tool is for educational purposes only.

It does not replace medical or psychological diagnosis.
"""
)

# ============================================
# MAIN TITLE
# ============================================

st.title("🧠 AI Mental Health Support Analyzer")

st.markdown(
    "### Understand emotional well-being using AI and NLP"
)

st.markdown("---")

# ============================================
# TABS
# ============================================

tab1, tab2, tab3 = st.tabs(
    ["📝 Analyze Mental Health", "📊 Dashboard", "📘 About Project"]
)

# ============================================
# TAB 1 - ANALYSIS
# ============================================

with tab1:

    st.subheader("Enter your thoughts")

    user_input = st.text_area(
        "How are you feeling today?",
        height=150,
        placeholder="Example: I feel stressed and anxious about my work and future"
    )

    col1, col2 = st.columns(2)

    analyze = col1.button("🔍 Analyze")
    clear = col2.button("🧹 Clear")

    if clear:
        st.rerun()

    if analyze:

        if user_input.strip() == "":
            st.warning("⚠ Please enter your thoughts")
        else:

            try:

                # =====================================
                # MODEL PREDICTION
                # =====================================

                prediction_encoded = model.predict([user_input])[0]

                label = encoder.inverse_transform(
                    [prediction_encoded]
                )[0]

                # =====================================
                # CONFIDENCE
                # =====================================

                try:
                    probabilities = model.predict_proba([user_input])
                    confidence = float(np.max(probabilities) * 100)
                except:
                    confidence = 80.0

                confidence_text = interpret_confidence(confidence)

                # =====================================
                # EMOTION MAPPING
                # =====================================

                emotion = map_emotion(label)

                risk, severity = calculate_risk(label, confidence)

                suggestions = suggest_help(label)

                helpline = get_helpline()

                emergency = detect_emergency(user_input)

                # =====================================
                # RESULT SECTION
                # =====================================

                st.markdown("---")
                st.subheader("🧾 Prediction Result")

                if label.lower() == "depressed":
                    st.error("🔴 Depression Detected")

                elif label.lower() == "stress":
                    st.warning("🟠 Stress Detected")

                elif label.lower() == "anxiety":
                    st.warning("🟡 Anxiety Detected")

                else:
                    st.success("🟢 Normal Emotional State")

                # =====================================
                # METRICS
                # =====================================

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Emotion", emotion["emotion"])

                col2.metric("Risk Level", risk)

                col3.metric("Severity", severity)

                col4.metric("Confidence", f"{confidence:.2f}%")

                st.write("Confidence Interpretation:", confidence_text)

                st.markdown("---")

                # =====================================
                # EMERGENCY ALERT
                # =====================================

                if emergency:

                    st.error(
                        "🚨 Emergency Detected. Please seek immediate help."
                    )

                # =====================================
                # SUGGESTIONS
                # =====================================

                st.subheader("💡 Suggested Support")

                for s in suggestions:
                    st.write("✔", s)

                st.markdown("---")

                # =====================================
                # HELPLINE
                # =====================================

                st.subheader("📞 Helpline Support")

                for h in helpline["helplines"]:

                    st.info(
                        f"{h['name']} : {h['number']} ({h['availability']})"
                    )

                # =====================================
                # SAVE HISTORY
                # =====================================

                new_record = pd.DataFrame({

                    "text": [user_input],
                    "prediction": [label],
                    "confidence": [confidence],
                    "risk": [risk],
                    "severity": [severity],
                    "time": [datetime.now()]
                })

                if os.path.exists(HISTORY_PATH):

                    old = pd.read_csv(HISTORY_PATH)

                    updated = pd.concat(
                        [old, new_record],
                        ignore_index=True
                    )

                else:
                    updated = new_record

                updated.to_csv(HISTORY_PATH, index=False)

                st.success("✅ Result saved to dashboard")

            except Exception as e:
                st.error("❌ Prediction failed")
                st.write(e)

# ============================================
# TAB 2 - DASHBOARD
# ============================================

with tab2:

    st.subheader("📊 Prediction Dashboard")

    if os.path.exists(HISTORY_PATH):

        history = pd.read_csv(HISTORY_PATH)

        st.write("Total Predictions:", len(history))

        st.dataframe(history)

        st.markdown("---")

        st.subheader("📈 Label Distribution")

        label_counts = history["prediction"].value_counts()

        st.bar_chart(label_counts)

        st.markdown("---")

        st.subheader("📉 Confidence Trend")

        st.line_chart(history["confidence"])

        st.markdown("---")

        csv = history.to_csv(index=False).encode()

        st.download_button(
            "⬇ Download Report",
            csv,
            "mental_health_report.csv",
            "text/csv"
        )

    else:
        st.info("No history available yet")

# ============================================
# TAB 3 - ABOUT
# ============================================

with tab3:

    st.subheader("📘 Project Overview")

    st.write(
        """
This project is an AI-based mental health support system
built using NLP and Machine Learning.

It predicts emotional state and provides support suggestions.
"""
    )

    st.markdown("---")

    st.subheader("⚙ Workflow")

    st.write(
        """
Dataset → Cleaning → TF-IDF → ML Model → Prediction → Streamlit UI
"""
    )

    st.markdown("---")

    st.subheader("🧰 Technologies")

    tech = pd.DataFrame({

        "Technology": [
            "Python",
            "Streamlit",
            "Scikit-learn",
            "Machine Learning",
            "TF-IDF",
            "NLP",
            "Pandas"
        ]
    })

    st.table(tech)

    st.markdown("---")

    st.subheader("🚀 Future Enhancements")

    st.write(
        """
Voice Input

Chatbot

Mobile App

Real-time Counseling

Emotion Graph

Multi-language Support
"""
    )
