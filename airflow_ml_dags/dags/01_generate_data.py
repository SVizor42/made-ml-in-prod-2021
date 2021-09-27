from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constants import DEFAULT_ARGS, RAW_DATA_PATH, HOST_DATA_DIR


with DAG(
        dag_id="01_generate_data",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(7),
) as dag:
    generate = DockerOperator(
        image="airflow-generate",
        command=f"--output_dir {RAW_DATA_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-generate",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"]
    )

    notify = BashOperator(
        task_id="generate-notify",
        bash_command="echo 'New package of data were generated.'"
    )

    generate >> notify
