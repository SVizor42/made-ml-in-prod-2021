Airflow project for ML in Production course
==============================
Prerequisites
------------
Ensure that your `airflow` version is at least 2.0.1 and `apache-airflow-providers-docker` was installed. 

Configure `airflow` parameters in [`data/constants.py`](https://github.com/SVizor42/made-ml-in-prod-2021/blob/homework3/airflow_ml_dags/dags/constants.py). Set paths to data folders in environment variables in [`docker-compose.yml`](https://github.com/SVizor42/made-ml-in-prod-2021/blob/homework3/airflow_ml_dags/docker-compose.yml).

Usage
------------
### Run Airflow
From root directory containing `docker-compose.yml`:
~~~bash
# Unix host
export MAIL_UID=your_gmail_name@gmail.com
export MAIL_PWD=your_gmail_password
export HOST_DATA_DIR=$(pwd)/data
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker-compose up --build

# Windows host
set MAIL_UID=your_gmail_name@gmail.com
set MAIL_PWD=your_gmail_password
set HOST_DATA_DIR=%cd%/data
python -c "import os; from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY);" > tmp.txt
set /P FERNET_KEY=<tmp.txt
del tmp.txt
docker-compose up --build
~~~
Next go to http://localhost:8080/, authorize using `login=admin` and `password=admin` and unpause all DAGs. After completion of `01_generate_data` and `02_train_pipeline` DAGs, open Admin->Variables section and add the variable with name `model` and value that is the date of one previously trained model (use `YYYY-MM-DD` format).
### Stop Airflow
~~~bash
docker-compose down
docker system prune  # yes
docker volume prune  # yes
docker network prune  # yes
~~~
### Test DAGs
To test the DAGs use `pytest` command from the root directory:
~~~bash
pytest -v .
~~~