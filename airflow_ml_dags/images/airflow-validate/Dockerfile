FROM airflow-ml-base:latest

COPY validate.py /validate.py
COPY requirements.txt /build/requirements.txt
RUN pip3 install --no-cache-dir -r /build/requirements.txt

ENTRYPOINT ["python", "validate.py"]