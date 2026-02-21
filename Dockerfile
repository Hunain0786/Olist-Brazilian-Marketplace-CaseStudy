FROM python:3.10-slim

WORKDIR /app

COPY src/requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

RUN python src/train_model.py && \
    python src/train_seller_models.py && \
    rm -rf /root/.cache/kagglehub

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]