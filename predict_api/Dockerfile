FROM python:3.8

WORKDIR /app

COPY predict_api/requirements.txt ./
RUN pip install -r requirements.txt

COPY predict_api/src ./
COPY configs/config.py ./

EXPOSE 8000
ENTRYPOINT [ "uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000" ]
