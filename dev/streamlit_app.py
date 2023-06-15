import streamlit as st

from configs.config import config
from scripts.predict_pipeline.predict import PredictPipeline

st.set_page_config(
    page_title="YT views predict",
    page_icon="🐙",
)

st.title("YouTube views prediction")

@st.cache_resource()
def load_predict_pipeline():
    a = PredictPipeline()
    from time import sleep
    sleep(5)
    return a

predict_pipeline = load_predict_pipeline()

with st.form(key="my_form"):
    channel_name = st.selectbox(
        "Channel name:",
        ["Other", "BT Sport", "CNN", "CGP Grey"],
    )
    category = st.selectbox(
        "Video category:",
        ["Entertainment", "Sports", "Education", "News & Politics"],
    )
    payload = {
        "channel": channel_name,
        "category": category,
    }
    submitted = st.form_submit_button("Predict")

    st.write("Predicted number of views:")

    if submitted:
        if payload["channel"] == "Other":
            payload["channel"] = "unk"
        if payload["category"] == "Other":
            payload["category"] = "unk"
        preds = predict_pipeline.predict(payload)
        st.info(f"{int(preds):.0f}")
