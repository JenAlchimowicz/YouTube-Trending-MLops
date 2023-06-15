import requests
import streamlit as st
from aws_utils import retrieve_item_list
from config import config
from PIL import Image

st.set_page_config(
    page_title="YT views predict",
    page_icon="🐙",
)

st.title("YouTube views prediction")

@st.cache_data(ttl=3600)
def retrieve_item_lists():
    possible_channels = retrieve_item_list(config.s3_bucket_name, config.s3_item_list_dir + "/channel_list.parquet")
    possible_categories = retrieve_item_list(config.s3_bucket_name, config.s3_item_list_dir + "/categories_list.parquet")
    return possible_channels, possible_categories

possible_channels, possible_categories = retrieve_item_lists()

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
        "allo\n"
        "contact\n"  # noqa: COM812
    )
    st.markdown("---")
    my_expander = st.expander(label="Are you Nev?")
    with my_expander:
        clicked = st.button("Yes")
        if clicked:
            st.balloons()
            image = Image.open("nev.png")
            st.image(image)

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
        preds = requests.post("http://yt-trending-api-445804270.eu-west-1.elb.amazonaws.com/", json=payload, timeout=5)
        st.info(f"{preds.json():,.0f}")
