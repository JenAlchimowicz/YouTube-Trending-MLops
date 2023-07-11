FROM python:3.8

WORKDIR /app

COPY feature_store_update/requirements.txt ./
RUN pip install -r requirements.txt

COPY feature_store_update/src ./
COPY configs/config.py ./

ENTRYPOINT [ "python", "main_feature_store_update.py" ]
