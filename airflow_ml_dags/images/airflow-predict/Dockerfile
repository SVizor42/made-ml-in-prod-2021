FROM airflow-ml-base:latest

COPY predict.py /predict.py
COPY requirements.txt /build/requirements.txt
RUN pip3 install --no-cache-dir -r /build/requirements.txt

ENTRYPOINT ["python", "predict.py"]