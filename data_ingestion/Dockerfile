FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update &&\
    apt-get install -y python3 python3-pip unzip &&\
    pip3 install awscli --upgrade &&\
    pip install kaggle &&\
    mkdir data

COPY data_ingestion/src ./

ENTRYPOINT [ "./data_download.sh" ]
