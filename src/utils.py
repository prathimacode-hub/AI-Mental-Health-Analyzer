"""
AI Mental Health Support Analyzer - Utility Module

Author: Prathima
Project: AI Mental Health Streamlit App

This module provides:

1. Text Cleaning
2. Emotion Mapping
3. Suggestion Engine
4. Risk & Severity Calculation
5. Emergency Detection
6. Helpline Service
7. Confidence Interpretation
8. Response Formatter
9. Constants & Config
"""

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import re
import string
from datetime import datetime
from typing import Dict, List, Tuple

# ==========================================
# CONSTANTS
# ==========================================

EMOTION_MAP = {

    "depressed": {
        "emotion": "Sadness",
        "emoji": "😔",
        "color": "red",
        "risk": "High"
    },

    "stress": {
        "emotion": "Stress",
        "emoji": "😟",
        "color": "orange",
        "risk": "Medium"
    },

    "anxiety": {
        "emotion": "Anxiety",
        "emoji": "😰",
        "color": "yellow",
        "risk": "Medium"
    },

    "normal": {
        "emotion": "Positive",
        "emoji": "😊",
        "color": "green",
        "risk": "Low"
    }
}

# ==========================================
# SUGGESTIONS
# ==========================================

SUGGESTIONS = {

    "depressed": [

        "Talk to a trusted friend or family member",
        "Consult a mental health professional",
        "Maintain a proper sleep routine",
        "Practice mindfulness meditation",
        "Spend time outdoors",
        "Avoid isolation",
        "Write your thoughts in a journal"
    ],

    "stress": [

        "Take short breaks during work",
        "Practice deep breathing exercises",
        "Organize your daily tasks",
        "Exercise regularly",
        "Listen to calming music",
        "Drink enough water"
    ],

    "anxiety": [

        "Try grounding techniques",
        "Practice meditation",
        "Avoid overthinking",
        "Stay hydrated",
        "Talk to someone you trust",
        "Take slow deep breaths"
    ],

    "normal": [

        "Maintain a positive mindset",
        "Stay physically active",
        "Help others",
        "Practice gratitude",
        "Keep social connections strong"
    ]
}

# ==========================================
# HELPLINE DATA
# ==========================================

HELPLINE_DATA = {

    "country": "India",

    "helplines": [

        {
            "name": "Kiran Mental Health Helpline",
            "number": "1800-599-0019",
            "availability": "24/7"
        },

        {
            "name": "National Tele Mental Health",
            "number": "9152987821",
            "availability": "24/7"
        },

        {
            "name": "Emergency Services",
            "number": "112",
            "availability": "24/7"
        }
    ]
}

# ==========================================
# EMERGENCY KEYWORDS
# ==========================================

EMERGENCY_KEYWORDS = [

    "suicide",
    "kill myself",
    "end my life",
    "want to die",
    "self harm",
    "hurt myself",
    "no reason to live",
    "hopeless",
    "die today"
]

# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text: str) -> str:
    """
    Clean user input text
    """

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = text.strip()

    return text


# ==========================================
# EMOTION MAPPING
# ==========================================

def map_emotion(label: str) -> Dict:
    """
    Map label to emotion
    """

    label = label.lower()

    return EMOTION_MAP.get(label, EMOTION_MAP["normal"])


# ==========================================
# SUGGESTION ENGINE
# ==========================================

def suggest_help(label: str) -> List[str]:
    """
    Return suggestions based on label
    """

    label = label.lower()

    return SUGGESTIONS.get(label, SUGGESTIONS["normal"])


# ==========================================
# RISK & SEVERITY
# ==========================================

def calculate_risk(
    label: str,
    confidence: float
) -> Tuple[str, str]:
    """
    Calculate risk and severity
    """

    label = label.lower()

    if label == "depressed":
        risk = "High"

    elif label in ["stress", "anxiety"]:
        risk = "Medium"

    else:
        risk = "Low"

    if confidence >= 90:
        severity = "Severe"

    elif confidence >= 75:
        severity = "Moderate"

    else:
        severity = "Mild"

    return risk, severity


# ==========================================
# EMERGENCY DETECTION
# ==========================================

def detect_emergency(text: str) -> bool:
    """
    Detect emergency suicidal intent
    """

    text = text.lower()

    for word in EMERGENCY_KEYWORDS:
        if word in text:
            return True

    return False


# ==========================================
# HELPLINE SERVICE
# ==========================================

def get_helpline() -> Dict:
    """
    Return helpline data
    """

    return HELPLINE_DATA


# ==========================================
# CONFIDENCE INTERPRETATION
# ==========================================

def interpret_confidence(confidence: float) -> str:
    """
    Interpret confidence score
    """

    if confidence >= 90:
        return "Very High Confidence"

    elif confidence >= 75:
        return "High Confidence"

    elif confidence >= 60:
        return "Moderate Confidence"

    else:
        return "Low Confidence"


# ==========================================
# RESPONSE FORMATTER
# ==========================================

def format_response(
    text: str,
    label: str,
    confidence: float
) -> Dict:
    """
    Build structured response
    """

    emotion = map_emotion(label)

    risk, severity = calculate_risk(
        label,
        confidence
    )

    suggestions = suggest_help(label)

    helpline = get_helpline()

    emergency = detect_emergency(text)

    response = {

        "input_text": text,

        "prediction": label,

        "confidence": confidence,

        "emotion": emotion,

        "risk": risk,

        "severity": severity,

        "suggestions": suggestions,

        "helpline": helpline,

        "emergency": emergency,

        "timestamp": str(datetime.now())
    }

    return response
