"""
AI Mental Health Support Analyzer - Training Pipeline

Author: Prathima
Project: AI Mental Health Streamlit App

Features:
- Dataset Loading
- Text Cleaning
- Label Encoding
- Train/Test Split
- TF-IDF Vectorization
- Machine Learning Model
- Pipeline Creation
- Model Evaluation
- Classification Report
- Confusion Matrix
- Model Saving
- Encoder Saving
- Directory Creation
- Logging
- Error Handling
- Production Ready
"""

# ============================================
# IMPORT LIBRARIES
# ============================================

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import (
    DATA_PATH,
    MODEL_PATH,
    TFIDF_PATH,
    ENCODER_PATH,
    PROCESSED_DATA_PATH,
    LOG_PATH
)

# ============================================
# CREATE MODEL DIRECTORY BEFORE LOGGING
# ============================================

MODEL_DIR = os.path.dirname(MODEL_PATH)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ============================================
# LOGGING
# ============================================

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================
# CREATE MODEL DIRECTORY BEFORE LOGGING
# ============================================

MODEL_DIR = os.path.dirname(MODEL_PATH)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ============================================
# LOGGING
# ============================================

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# ============================================
# CREATE MODEL DIRECTORY
# ============================================

MODEL_DIR = os.path.dirname(MODEL_PATH)
HISTORY_PATH = os.path.join(MODEL_DIR, "history.csv")


def create_directories():

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print("Model directory created")
        logging.info("Model directory created")


# ============================================
# LOAD DATA
# ============================================

def load_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded:", df.shape)
    logging.info(f"Dataset loaded {df.shape}")

    return df


# ============================================
# VALIDATE DATA
# ============================================

def validate_data(df):

    df.columns = df.columns.str.lower().str.strip()

    print("Columns:", df.columns)

    if "clean_text" not in df.columns:
        raise ValueError("clean_text column missing")

    if "is_depression" not in df.columns:
        raise ValueError("is_depression column missing")

    df = df.dropna()

    df = df.rename(columns={
        "clean_text": "text",
        "is_depression": "label"
    })

    df["text"] = df["text"].astype(str)

    print("Data validation complete")
    logging.info("Data validation complete")

    return df


# ============================================
# LABEL ENCODING
# ============================================

def encode_labels(df):

    encoder = LabelEncoder()

    df["label_encoded"] = encoder.fit_transform(df["label"])

    print("Labels:", list(encoder.classes_))
    logging.info(f"Labels {list(encoder.classes_)}")

    return df, encoder


# ============================================
# SPLIT DATA
# ============================================

def split_data(df):

    X = df["text"]
    y = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Data split complete")
    logging.info("Data split complete")

    return X_train, X_test, y_train, y_test


# ============================================
# BUILD PIPELINE
# ============================================

def build_pipeline():

    pipeline = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2)
            )
        ),

        (
            "model",
            LogisticRegression(max_iter=1000)
        )
    ])

    print("Pipeline created")
    logging.info("Pipeline created")

    return pipeline


# ============================================
# TRAIN MODEL
# ============================================

def train_model(pipeline, X_train, y_train):

    pipeline.fit(X_train, y_train)

    print("Model training complete")
    logging.info("Model training complete")

    return pipeline


# ============================================
# EVALUATE MODEL
# ============================================

def evaluate_model(model, X_test, y_test, encoder):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", accuracy)

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, predictions))

    logging.info(f"Accuracy {accuracy}")


# ============================================
# SAVE MODEL
# ============================================

def save_model(model, encoder):

    tfidf = model.named_steps["tfidf"]

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(tfidf, open(TFIDF_PATH, "wb"))
    pickle.dump(encoder, open(ENCODER_PATH, "wb"))

    print("Model saved")
    logging.info("Model saved")


# ============================================
# SAVE PROCESSED DATA
# ============================================

def save_processed_data(df):

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Processed data saved")
    logging.info("Processed data saved")


# ============================================
# CREATE HISTORY FILE
# ============================================

def create_history():

    if not os.path.exists(HISTORY_PATH):

        history = pd.DataFrame(columns=[

            "text",
            "prediction",
            "confidence",
            "risk",
            "severity",
            "time"
        ])

        history.to_csv(HISTORY_PATH, index=False)

        print("History file created")
        logging.info("History file created")


# ============================================
# MAIN
# ============================================

def main():

    print("Training started")
    logging.info("Training started")

    create_directories()

    df = load_dataset()

    df = validate_data(df)

    df, encoder = encode_labels(df)

    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline()

    model = train_model(pipeline, X_train, y_train)

    evaluate_model(model, X_test, y_test, encoder)

    save_model(model, encoder)

    save_processed_data(df)

    create_history()

    print("Training completed successfully")
    logging.info("Training completed")


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()
