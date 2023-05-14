import streamlit as st
from configs.config import config
from scripts.predict_pipeline.predict import PredictPipeline

st.set_page_config(
    page_title="YT views predict",
    # page_icon=""
)

st.title("YouTube views prediction")

@st.cache_data
def load_predict_pipeline():
    return PredictPipeline(
        checkpoint_path=config.model_path,
        train_df_path=config.train_set_path,
        feature_store_path=config.feature_store_path,
        ohe_category=config.artifacts_dir.joinpath("ohe_encoder_category.joblib"),
    )

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
        preds = predict_pipeline.predict(payload)
        st.info(f"{int(preds):.2f}")

# st.info("asd")
