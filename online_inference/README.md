ML service for ML in Production course
==============================
Prerequisites
------------
First of all, follow the tips in loading dataset and training the model in [ml_project](https://github.com/SVizor42/made-ml-in-prod-2021/tree/homework1/ml_project). Then put the trained model into the `models` directory.

Installation
------------
### Build docker image
```bash
docker build -t svizor42/online_inference:v1
```
### Load from DockerHub
```bash
docker pull svizor42/online_inference:v1
```

Usage
------------
### Run inference
From docker:
```bash
docker run --rm -p 8000:8000 svizor42/online_inference:v1
```
or directly:
```bash
PATH_TO_MODEL=models/model.pkl uvicorn app:app --host 0.0.0.0 --port 8000
```
and make requests from another terminal:
```bash
python src/make_request.py --dataset_path=data/data.csv
```
### Run tests
```bash
pip install -q pytest pytest-cov
python -m pytest . -v --cov
```

Project Structure
------------

    ├── data
    │   └── data.csv            <- The data needed to make test requests.
    │
    ├── models                  <- Trained and serialized models and pipelines.
    │
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module.
    │   ├── entities.py
    │   └── make_request.py
    │
    ├── tests                   <- Code to test project modules and pipelines.
    │   └── test_app.py
    │
    ├── app.py                  <- FastAPI application code.
    │
    ├── Dockerfile              <- Docker file to build image.
    │
    ├── README.md               <- The top-level README for developers using this project.
    │
    └── requirements.txt        <- The requirements file for reproducing the analysis environment.