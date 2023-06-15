FROM python:3.8

WORKDIR /app

COPY web_endpoint/requirements.txt ./
RUN pip install -r requirements.txt

COPY web_endpoint/src ./
COPY configs/config.py ./

ENTRYPOINT [ "streamlit", "run", "streamlit_app.py", "--server.port", "8501" ]
