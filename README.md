# Diabetes Prediction: automated MLOps pipeline

PMLDL Assignment 1. A simple automated pipeline that cleans data, trains a model,
and deploys it as an API with a web application. Everything runs automatically every 5 minutes.

**Dataset:** [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
(768 rows, 8 numeric features, target `Outcome`: 1 = diabetes, 0 = no diabetes).

## Pipeline

| Stage | What it does | Code | Output |
|---|---|---|---|
| 1. Data engineering | Loads raw csv, replaces impossible zeros with NaN, removes outliers (IQR rule), fills missing values with the median, splits 80/20 | `code/datasets/data_engineering.py` | `data/processed/train.csv`, `data/processed/test.csv` |
| 2. Model engineering | Scales features (StandardScaler), trains Logistic Regression, evaluates on test data, logs params, metrics and model to MLflow | `code/models/train_model.py` | `models/model.pkl`, `models/metrics.json`, `mlflow.db` |
| 3. Deployment | Builds and starts two Docker containers with docker compose: FastAPI (model API) and Streamlit (web app) | `code/deployment/` | API on port 8000, app on port 8501 |

The scheduler `code/pipeline/run_pipeline.py` runs the three stages one after another and repeats them every 5 minutes.
If a stage fails, the next stages are skipped and the previously deployed app keeps working.

## Repository structure

```
├── code
│   ├── datasets/data_engineering.py      # Stage 1
│   ├── models/train_model.py             # Stage 2
│   ├── deployment
│   │   ├── api/                          # FastAPI + Dockerfile
│   │   ├── app/                          # Streamlit + Dockerfile
│   │   └── docker-compose.yml
│   └── pipeline/run_pipeline.py          # automation (every 5 minutes)
├── data
│   ├── raw/diabetes.csv
│   └── processed/                        # train.csv, test.csv
├── models/                               # model.pkl, metrics.json
├── logs/                                 # pipeline.log (created automatically)
├── requirements.txt
└── README.md
```

## How to run

Requirements: Python 3.10+, Docker Desktop (must be running).

```bash
# 1. create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 2. install libraries
pip install -r requirements.txt

# 3. start the automated pipeline (runs every 5 minutes, stop with Ctrl+C)
python code/pipeline/run_pipeline.py
```

To run the pipeline only one time:

```bash
python code/pipeline/run_pipeline.py --once
```

The raw data is already in `data/raw/diabetes.csv`.

## Access

| Service | URL |
|---|---|
| Web application (Streamlit) | http://localhost:8501 |
| Model API docs (FastAPI) | http://localhost:8000/docs |
| MLflow UI | run `mlflow ui --backend-store-uri sqlite:///mlflow.db`, then open http://localhost:5000 |

Enter patient data in the web application and press **Predict**. The application sends a request to the API
(`POST /predict`), and the API returns the prediction and its probability.

To stop the containers:

```bash
docker compose -f code/deployment/docker-compose.yml down
```

## Results

Test metrics of the model are saved to `models/metrics.json` after every run and logged to MLflow
(accuracy, precision, recall, F1, ROC-AUC).
