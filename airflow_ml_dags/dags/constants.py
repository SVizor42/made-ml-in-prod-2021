from datetime import timedelta
from airflow.models import Variable

DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "email_on_failure": True,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

AIRFLOW_RAW_DATA_PATH = "/opt/airflow/data/raw/{{ ds }}"
RAW_DATA_PATH = "/data/raw/{{ ds }}"
PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
SPLITTED_DATA_PATH = "/data/processed/{{ ds }}"
PREDICTIONS_PATH = "/data/predictions/{{ ds }}"
MODEL_PATH = "/data/models/{{ ds }}"
HOST_DATA_DIR = Variable.get("HOST_DATA_DIR")
