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

import os
import pandas as pd
import numpy as np
import pickle
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================
# LOGGING CONFIG
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================
# PATHS
# ============================================

DATASET_PATH = "dataset/mental_health.csv"
MODEL_DIR = "model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mental_health_model.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "history.csv"
)

# ============================================
# CREATE FOLDERS
# ============================================

def create_directories():

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logging.info("Model directory created")

# ============================================
# LOAD DATA
# ============================================

def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "Dataset not found in dataset/mental_health.csv"
        )

    df = pd.read_csv(DATASET_PATH)

    logging.info(f"Dataset loaded: {df.shape}")

    return df

# ============================================
# VALIDATE DATA
# ============================================

def validate_data(df):

    required_columns = ["text", "label"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"{col} column missing in dataset"
            )

    df = df.dropna()

    df["text"] = df["text"].astype(str)

    logging.info("Data validation complete")

    return df

# ============================================
# LABEL ENCODING
# ============================================

def encode_labels(df):

    encoder = LabelEncoder()

    df["label_encoded"] = encoder.fit_transform(
        df["label"]
    )

    logging.info(
        f"Labels: {list(encoder.classes_)}"
    )

    return df, encoder

# ============================================
# TRAIN TEST SPLIT
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
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    logging.info("Pipeline created")

    return pipeline

# ============================================
# TRAIN MODEL
# ============================================

def train_model(pipeline, X_train, y_train):

    pipeline.fit(X_train, y_train)

    logging.info("Model training complete")

    return pipeline

# ============================================
# EVALUATION
# ============================================

def evaluate_model(model, X_test, y_test, encoder):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    logging.info(f"Accuracy: {accuracy}")

    print("\nAccuracy:", accuracy)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_
        )
    )

    print("\nConfusion Matrix:\n")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

# ============================================
# SAVE MODEL
# ============================================

def save_model(model, encoder):

    pickle.dump(
        model,
        open(MODEL_PATH, "wb")
    )

    pickle.dump(
        encoder,
        open(ENCODER_PATH, "wb")
    )

    logging.info("Model saved")

# ============================================
# CREATE HISTORY FILE
# ============================================

def create_history_file():

    if not os.path.exists(HISTORY_PATH):

        df = pd.DataFrame(columns=[

            "text",
            "prediction",
            "confidence",
            "risk",
            "severity",
            "time"
        ])

        df.to_csv(HISTORY_PATH, index=False)

        logging.info("History file created")

# ============================================
# MAIN FUNCTION
# ============================================

def main():

    logging.info("Training started")

    create_directories()

    df = load_dataset()

    df = validate_data(df)

    df, encoder = encode_labels(df)

    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline()

    model = train_model(
        pipeline,
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        encoder
    )

    save_model(
        model,
        encoder
    )

    create_history_file()

    logging.info("Training completed successfully")

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()
