FROM python:3.8

WORKDIR /app

COPY data_transformation/requirements.txt ./
RUN pip install -r requirements.txt

COPY data_transformation/src ./
COPY configs/config.py ./

ENTRYPOINT [ "python", "main_data_transformation.py" ]
