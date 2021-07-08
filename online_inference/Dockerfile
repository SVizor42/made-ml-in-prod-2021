FROM python:3.7-slim-stretch
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src /src
COPY app.py /app.py
COPY models /models

WORKDIR .

ENV PATH_TO_MODEL="models/model.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]