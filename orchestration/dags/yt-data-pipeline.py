from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator


def _start():
    print("Is this working for once")


with DAG(
    "yt-data-pipeline",
    start_date=datetime(2023, 9, 24),
    schedule_interval="@weekly",
    catchup=False,
) as dag:

    test_task = PythonOperator(
        task_id="task1",
        python_callable=_start,
    )

    data_ingestion = EcsRunTaskOperator(
        task_id="data-ingestion",
        aws_conn_id="aws_admin",
        region="eu-west-1",
        task_definition="yt-trending-data-ingestion",
        cluster="yt-trending-api",
        overrides={},
        launch_type="FARGATE",
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ["subnet-0d42b831d6214a74a", "subnet-0a729e5b70f8218f7", "subnet-02bac66c89cdc0b7a"],
                "securityGroups": ["sg-02d5329db86179953"],
                "assignPublicIp": "ENABLED",
            },
        },
    )

    data_transformation = EcsRunTaskOperator(
        task_id="data-transformation",
        aws_conn_id="aws_admin",
        region="eu-west-1",
        task_definition="yt-trending-data-transformation",
        cluster="yt-trending-api",
        overrides={},
        launch_type="FARGATE",
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ["subnet-0d42b831d6214a74a", "subnet-0a729e5b70f8218f7", "subnet-02bac66c89cdc0b7a"],
                "securityGroups": ["sg-02d5329db86179953"],
                "assignPublicIp": "ENABLED",
            },
        },
    )

    feature_store_update = EcsRunTaskOperator(
        task_id="feature-store-update",
        aws_conn_id="aws_admin",
        region="eu-west-1",
        task_definition="yt-trending-feature-store-update",
        cluster="yt-trending-api",
        overrides={},
        launch_type="FARGATE",
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ["subnet-0d42b831d6214a74a", "subnet-0a729e5b70f8218f7", "subnet-02bac66c89cdc0b7a"],
                "securityGroups": ["sg-02d5329db86179953"],
                "assignPublicIp": "ENABLED",
            },
        },
    )

    training = EcsRunTaskOperator(
        task_id="training",
        aws_conn_id="aws_admin",
        region="eu-west-1",
        task_definition="yt-trending-training",
        cluster="yt-trending-api",
        overrides={},
        launch_type="FARGATE",
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ["subnet-0d42b831d6214a74a", "subnet-0a729e5b70f8218f7", "subnet-02bac66c89cdc0b7a"],
                "securityGroups": ["sg-02d5329db86179953"],
                "assignPublicIp": "ENABLED",
            },
        },
    )

    test_task >> \
    data_ingestion >> \
    data_transformation >> \
    feature_store_update >> \
    training
