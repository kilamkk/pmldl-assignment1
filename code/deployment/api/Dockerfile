FROM python:3.12-slim

WORKDIR /app

# install libraries first (Docker caches this layer)
COPY code/deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the API code and the trained model from Stage 2
COPY code/deployment/api/main.py .
COPY models/model.pkl .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
