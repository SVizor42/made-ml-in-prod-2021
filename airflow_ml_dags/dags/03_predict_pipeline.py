from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constants import (
    AIRFLOW_RAW_DATA_PATH,
    DEFAULT_ARGS,
    HOST_DATA_DIR,
    PREDICTIONS_PATH,
    RAW_DATA_PATH,
)


with DAG(
        "03_predict_pipeline",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(7),
) as dag:
    wait_for_inf_data = FileSensor(
        task_id="wait-for-inference-data",
        poke_interval=10,
        retries=5,
        filepath=f"{AIRFLOW_RAW_DATA_PATH}/data.csv",
        mode="poke"
    )

    wait_for_transformer = FileSensor(
        task_id="wait-for-transformer",
        poke_interval=10,
        retries=5,
        filepath="data/models/{{ var.value.model }}/transformer.pkl",
        mode="poke"
    )

    wait_for_model = FileSensor(
        task_id="wait-for-model",
        poke_interval=10,
        retries=5,
        filepath="data/models/{{ var.value.model }}/model.pkl",
        mode="poke"
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input_dir {RAW_DATA_PATH} --output_dir {PREDICTIONS_PATH} "
                "--model_dir /data/models/{{ var.value.model }}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"]
    )

    notify = BashOperator(
        task_id="predict-notify",
        bash_command="echo 'Predictions were made and saved.'"
    )

    [wait_for_inf_data, wait_for_transformer, wait_for_model] >> predict >> notify
