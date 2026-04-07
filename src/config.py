import os

# =========================
# Base Directory
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# Dataset Path
# =========================

DATA_PATH = os.path.join(BASE_DIR, "dataset", "mental_health.csv")

# =========================
# Model Paths
# =========================

MODEL_PATH = os.path.join(BASE_DIR, "models", "mental_health_model.pkl")

TFIDF_PATH = os.path.join(BASE_DIR, "models", "tfidf.pkl")

ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "models", "processed_data.csv")

# =========================
# Logs
# =========================

LOG_PATH = os.path.join(BASE_DIR, "models", "training.log")

# =========================
# App Settings
# =========================

APP_TITLE = "AI Mental Health Support Analyzer"

APP_DESCRIPTION = "Detects emotions and provides mental health support suggestions"

