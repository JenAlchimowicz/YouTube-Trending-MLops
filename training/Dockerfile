FROM python:3.8

WORKDIR /app

COPY training/requirements.txt ./
RUN pip install -r requirements.txt

COPY training/src ./
COPY configs/config.py ./

ENTRYPOINT [ "python", "main_training.py" ]
