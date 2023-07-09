import streamlit as st
from aws_utils import retrieve_item_list
from config import config
from predict import PredictPipeline

st.set_page_config(
    page_title="YT views predict",
    page_icon="🐙",
)

st.title("YouTube views prediction")


@st.cache_resource(ttl=3600)
def retrieve_predict_pipeline():
    pp = PredictPipeline()
    return pp

@st.cache_data(ttl=3600)
def retrieve_item_lists():
    possible_channels = retrieve_item_list(config.s3_bucket_name, config.s3_item_list_dir + "/channel_list.parquet")
    possible_categories = retrieve_item_list(config.s3_bucket_name, config.s3_item_list_dir + "/categories_list.parquet")
    return possible_channels, possible_categories

possible_channels, possible_categories = retrieve_item_lists()
predict_pipeline = retrieve_predict_pipeline()

with st.sidebar:
    st.markdown(
        "# How to use\n"
        '1. Select the channel you want to make predictions for. If your channels in not in the list type in "unk" for unknown\n'  # noqa: E501
        "2. Select the category of the video\n"
        "3. Press **Predict**\n"  # noqa: COM812
    )
    st.markdown("---")
    st.markdown(
        "# About\n"
        "*The goal of this app is to demonstrate how to deploy ML apps on AWS.* \n\nThe problem itself - predicting "
        "YouTube views from just the channel name and video category - is rather trivial, and would usually be more "
        "complex in the real world. However, the methods of managing the ML lifecycle are very relevant and can be "
        "used to deploy real-world projects.\n"
        "\nRepo link [here](https://github.com/JenAlchimowicz/YouTube-Trending-MLops)"# noqa: COM812
    )
    st.markdown("---")
    st.markdown("# Social\n")
    column1, column2, column3 = st.columns(3)
    column1.markdown("[![Title](https://img.icons8.com/?size=70&id=44019&format=png)](https://www.linkedin.com/in/jen-alchimowicz)")
    column2.markdown("[![Title](https://img.icons8.com/?size=70&id=52539&format=png)](https://github.com/JenAlchimowicz)")
    column3.markdown("[![Title](https://img.icons8.com/?size=70&id=gU6bwZNC5TXf&format=png)](https://medium.com/@jedrzejalchimowicz)")

with st.form(key="my_form"):
    channel_name = st.selectbox("Channel name:", possible_channels)
    category = st.selectbox("Video category:", possible_categories)
    payload = {
        "channel": channel_name,
        "category": category,
    }
    submitted = st.form_submit_button("Predict")

    st.write("Predicted number of views:")
    if submitted:
        preds = predict_pipeline.predict(payload)
        st.info(f"{preds:,.0f}")
