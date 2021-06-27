ML project for ML in Production course
==============================
Dataset
------------
[Heart Disease UCI Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)

Train model
------------
```bash
python src/train_pipeline.py
```
Make predictions
------------
```bash
python src/predict_pipeline.py
```
Run tests
------------
```bash
pip install pytest pytest-cov
python -m pytest . -v --cov=src
```

Project Structure
------------

    ├── configs                     <- Configuration files for projects modules.
    │   ├── train_config.yaml
    │   └── predict_config.yaml
    │
    ├── data
    │   └── raw                     <- The original, immutable data dump.
    │
    ├── models                      <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks                   <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                               the creator's initials, and a short `-` delimited description, e.g.
    │                               `1.0-jqp-initial-data-exploration`.
    │
    ├── reports                     <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── src                         <- Source code for use in this project.
    │   ├── __init__.py             <- Makes src a Python module.
    │   │
    │   ├── data                    <- Scripts to download or generate data.
    │   │   └── make_dataset.py
    │   │
    │   ├── entities                <- Parameters for use in this project.
    │   │   ├── feature_params.py
    │   │   ├── split_params.py
    │   │   ├── train_params.py
    │   │   ├── train_pipeline_perams.py
    │   │   └── predict_pipeline_params.py
    │   │
    │   ├── features                <- Scripts to turn raw data into features for modeling.
    │   │   ├── build_features.py
    │   │   └── transformer.py
    │   │
    │   ├── models                  <- Scripts to train models and then use trained models to make
    │   │   │                       predictions.
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── predict_pipeline.py
    │   └── train_pipeline.py
    │
    ├── tests                       <- Code to test project modules and pipelines.
    │   ├── data
    │   │   └── test_make_dataset.py
    │   │
    │   ├── features
    │   │   ├── test_make_categorical_features.py
    │   │   ├── test_make_features.py
    │   │   └── test_transformer.py
    │   │
    │   ├── models
    │   │   └── test_train_model.py
    │   │
    │   ├── conftest.py
    │   ├── test_end2end_prediction.py
    │   ├── test_end2end_training.py
    │   └── train_data_sample.csv
    │
    ├── LICENSE
    ├── Makefile                    <- Makefile with commands like `make data` or `make train`.
    ├── README.md                   <- The top-level README for developers using this project.
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                               generated with `pip freeze > requirements.txt`.
    │
    ├── setup.py                    <- Makes project pip installable (pip install -e .) so src can be imported
    └── tox.ini                     <- tox file with settings for running tox; see tox.readthedocs.io.

