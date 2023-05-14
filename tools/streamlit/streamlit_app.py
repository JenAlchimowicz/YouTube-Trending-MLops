import streamlit as st

from ...configs.config import config
from ...scripts.predict_pipeline import PredictPipeline

"allo"
"whoo"

@st.cache_data
def load_predict_pipeline():
    return PredictPipeline(
        checkpoint_path=config.model_path,
        train_df_path=config.train_set_path,
        feature_store_path=config.feature_store_path,
        ohe_category=config.artifacts_dir.joinpath("ohe_encoder_category.joblib"),
    )


