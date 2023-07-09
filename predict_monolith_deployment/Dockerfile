FROM python:3.8

WORKDIR /app

COPY predict_monolith_deployment/requirements.txt ./
RUN pip install -r requirements.txt

COPY predict_monolith_deployment/src ./
COPY configs/config.py ./

ENTRYPOINT [ "streamlit", "run", "streamlit_app.py", "--server.port", "8501" ]
