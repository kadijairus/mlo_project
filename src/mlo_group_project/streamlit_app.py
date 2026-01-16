from __future__ import annotations

import requests
import streamlit as st

API_URL_DEFAULT = "http://127.0.0.1:8000/evaluate-csv"

st.set_page_config(page_title="Breast Cancer Evaluator", layout="centered")

st.title("Breast Cancer CSV Evaluation")
st.subheader("MLOps Group 5, 2026")
st.write("Upload a CSV dataset. The app sends it to the FastAPI backend for preprocessing + evaluation.")
# Dataset can be found at kaggle
st.write("Data can be found at: [Link to Kaggle data set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)")

api_url = st.text_input("FastAPI endpoint", value=API_URL_DEFAULT)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

col1, col2 = st.columns([1, 2])
with col1:
    run_eval = st.button("Evaluate", disabled=uploaded is None)

with col2:
    st.caption("IMPORTANT: Backend must be running: uvicorn mlo_group_project.api:app")

if run_eval:
    if uploaded is None:
        st.error("Please upload a CSV file.")
        st.stop()

    try:
        files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
        with st.spinner("Uploading and evaluating..."):
            r = requests.post(api_url, files=files, timeout=120)
    except requests.RequestException as e:
        st.error(f"Failed to reach API: {e}")
        st.stop()

    if r.status_code != 200:
        st.error(f"API error {r.status_code}")
        try:
            st.json(r.json())
        except Exception:
            st.text(r.text)
        st.stop()

    result = r.json()
    st.success("Evaluation complete")
    st.json(result)

    # Display key metrics
    if result.get("has_labels"):
        st.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
        st.write(f"Correct: {result['correct']} / {result['total']}")
    else:
        st.write(result.get("message", "No labels found."))
        st.write(f"Predicted positive: {result.get('predicted_positive')}")