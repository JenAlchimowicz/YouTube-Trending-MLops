#!/usr/bin/env bash

kaggle datasets download -d "rsrishav/youtube-trending-video-dataset" -f "GB_youtube_trending_data.csv" -p "data/raw_data"
unzip -o -d data/raw_data data/raw_data/GB_youtube_trending_data.csv.zip
rm data/raw_data/GB_youtube_trending_data.csv.zip
kaggle datasets download -d "rsrishav/youtube-trending-video-dataset" -f "GB_category_id.json" -p "data/raw_data"

aws s3 cp ./data/raw_data/GB_youtube_trending_data.csv s3://yt-trending-mlops/data/raw_data/
aws s3 cp ./data/raw_data/GB_category_id.json s3://yt-trending-mlops/data/raw_data/
