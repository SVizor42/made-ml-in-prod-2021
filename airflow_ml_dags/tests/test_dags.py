import pytest
import sys
from airflow.models import DagBag

sys.path.append("dags")


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder="dags/", include_examples=False)


def assert_dag_dict_equal(source, dag):
    assert dag.task_dict.keys() == source.keys()
    for task_id, downstream_list in source.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)


@pytest.mark.parametrize(
    "dag_id, num_tasks",
    [
        ("01_generate_data", 2),
        ("02_train_pipeline", 7),
        ("03_predict_pipeline", 5),
    ]
)
def test_dag_loaded(dag_bag, dag_id, num_tasks):
    dag = dag_bag.get_dag(dag_id=dag_id)
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == num_tasks


@pytest.mark.parametrize(
    "source, dag_id",
    [
        (
            {
                "docker-airflow-generate": ["generate-notify"],
                "generate-notify": [],
            },
            "01_generate_data",
        ),
        (
            {
                "wait-for-features": ["docker-airflow-preprocess"],
                "wait-for-target": ["docker-airflow-preprocess"],
                "docker-airflow-preprocess": ["docker-airflow-split"],
                "docker-airflow-split": ["docker-airflow-train"],
                "docker-airflow-train": ["docker-airflow-validate"],
                "docker-airflow-validate": ["train-notify"],
                "train-notify": [],
            },
            "02_train_pipeline",
        ),
        (
            {
                "wait-for-inference-data": ["docker-airflow-predict"],
                "wait-for-transformer": ["docker-airflow-predict"],
                "wait-for-model": ["docker-airflow-predict"],
                "docker-airflow-predict": ["predict-notify"],
                "predict-notify": [],
            },
            "03_predict_pipeline",
        ),
    ]
)
def test_dag_structure(dag_bag, source, dag_id):
    dag = dag_bag.get_dag(dag_id=dag_id)
    assert_dag_dict_equal(source, dag)
