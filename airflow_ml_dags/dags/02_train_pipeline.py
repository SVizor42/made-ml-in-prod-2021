from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constants import (
    AIRFLOW_RAW_DATA_PATH,
    DEFAULT_ARGS,
    HOST_DATA_DIR,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    SPLITTED_DATA_PATH,
    MODEL_PATH,
)


with DAG(
        "02_train_pipeline",
        default_args=DEFAULT_ARGS,
        schedule_interval="@weekly",
        start_date=days_ago(7),
) as dag:
    wait_for_features = FileSensor(
        task_id="wait-for-features",
        poke_interval=10,
        retries=5,
        filepath=f"{AIRFLOW_RAW_DATA_PATH}/data.csv",
        mode="poke"
    )

    wait_for_target = FileSensor(
        task_id="wait-for-target",
        poke_interval=10,
        retries=5,
        filepath=f"{AIRFLOW_RAW_DATA_PATH}/target.csv",
        mode="poke"
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input_dir {RAW_DATA_PATH} --output_dir {PROCESSED_DATA_PATH} \
                --model_dir {MODEL_PATH}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"]
    )

    split = DockerOperator(
        image="airflow-split",
        command=f"--input_dir {PROCESSED_DATA_PATH} --output_dir {SPLITTED_DATA_PATH}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"]
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--input_dir {SPLITTED_DATA_PATH} --model_dir {MODEL_PATH}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--input_dir {SPLITTED_DATA_PATH} --model_dir {MODEL_PATH}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"]
    )

    notify = BashOperator(
        task_id="train-notify",
        bash_command="echo 'Model was trained and validated.'"
    )

    [wait_for_features, wait_for_target] >> preprocess >> split >> train >> validate >> notify
