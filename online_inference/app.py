import logging
import os
import sys
import pickle
import pandas as pd
import uvicorn
from typing import List, Optional
from fastapi import FastAPI
from sklearn.pipeline import Pipeline

from src.entities import HeartDiseaseModel, HeartDiseaseResponse
from src.entities import MAIN_ENDPOINT_MSG


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


model: Optional[Pipeline] = None

app = FastAPI()


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_predict(
    data: List, features: List[str], model: Pipeline,
) -> List[HeartDiseaseResponse]:
    df = pd.DataFrame(data, columns=features)
    logger.info(f"Data shape: {df.shape}.")
    logger.info(f"Dataset:\n{df}")
    predicts = model.predict(df)

    return [
        HeartDiseaseResponse(id=int(id_), disease=int(disease_))
        for id_, disease_ in zip(df.index.values, predicts)
    ]


@app.get("/")
def main():
    return MAIN_ENDPOINT_MSG


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/health")
def health() -> bool:
    return f"Pipeline is ready: {model is not None}"


@app.get("/predict/", response_model=List[HeartDiseaseResponse])
def predict(request: HeartDiseaseModel):
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
