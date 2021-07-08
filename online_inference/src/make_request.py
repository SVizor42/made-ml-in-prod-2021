import numpy as np
import pandas as pd
import requests
import click


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=8000)
@click.option("--dataset_path", default="../data/data.csv")
def send_requests(host: str, port: int, dataset_path: str):
    data = pd.read_csv(dataset_path, sep=';')
    target = data["target"]
    data = data.drop("target", axis=1)
    request_features = list(data.columns)
    for i in range(data.shape[0]):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print(request_data)
        response = requests.get(
            f"http://{host}:{port}/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(f"Response code: {response.status_code}")
        print(f"Response: {response.json()[0]}, target: {target[i]}")


if __name__ == "__main__":
    send_requests()
