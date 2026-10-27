import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRatio",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "Roundness", "Compactness", "ShapeFactor1", "ShapeFactor2",
    "ShapeFactor3", "ShapeFactor4",
]
TARGET = "Class"

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")
st.title("Dry Bean Type Classification")
st.write(
    "Seven dry-bean varieties are predicted from 16 size and shape measurements. "
    "Upload the test CSV. The chosen saved model is loaded and scored. It is not retrained."
)

model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
uploaded = st.file_uploader("Upload test CSV", type=["csv"])

if uploaded is not None:
    data = pd.read_csv(uploaded)
    missing = [column for column in FEATURES + [TARGET] if column not in data.columns]
    if missing:
        st.error("These columns are missing: " + ", ".join(missing))
    else:
        model = joblib.load(MODEL_FILES[model_name])
        X = data[FEATURES]
        y_true = data[TARGET]
        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)
        else:
            y_score = softmax(model.decision_function(X), axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader(model_name)
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("AUC", f"{auc:.4f}")
        c3.metric("Precision", f"{precision:.4f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("Recall", f"{recall:.4f}")
        c5.metric("F1", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        labels = sorted(y_true.unique())
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Predicted bean type")
        ax.set_ylabel("Actual bean type")
        st.subheader("Confusion matrix")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification report")
        st.text(classification_report(y_true, y_pred))
