FROM python:3.9-slim
WORKDIR /app
COPY . /app/
RUN pip install --no-cache-dir pandas numpy scikit-learn mlflow
CMD ["python", "modelling.py"]