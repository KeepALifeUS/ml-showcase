# Multi-stage Docker build для Social Media Sentiment Analyzer
#  enterprise-grade containerization

FROM python:3.11-slim as base

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя
RUN useradd --create-home --shell /bin/bash app

# Рабочая директория
WORKDIR /app

# Копирование requirements
COPY requirements.txt pyproject.toml ./

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Загрузка ML моделей
FROM base as models

# Создание директории для моделей
RUN mkdir -p /app/models

# Python скрипт для загрузки моделей
RUN python -c "
import nltk
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# NLTK данные
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True) 
nltk.download('vader_lexicon', quiet=True)

# spaCy модель
import subprocess
subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)

# Transformers модели
models = [
    'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'nlptown/bert-base-multilingual-uncased-sentiment',
    'ProsusAI/finbert'
]

for model_name in models:
    print(f'Loading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Сохранение в кэш
    tokenizer.save_pretrained(f'/app/models/{model_name.replace(\"/\", \"_\")}')
    model.save_pretrained(f'/app/models/{model_name.replace(\"/\", \"_\")}')
    print(f'Cached {model_name}')

print('All models downloaded and cached!')
"

# Production image
FROM base as production

# Копирование кода
COPY src/ ./src/
COPY tests/ ./tests/
COPY --from=models /app/models ./models/

# Права доступа
RUN chown -R app:app /app
USER app

# Переменные окружения
ENV PYTHONPATH=/app
ENV ML_MODEL_CACHE_DIR=/app/models
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Экспозиция портов
EXPOSE 8004 9090

# Запуск приложения
CMD ["uvicorn", "src.api.rest_api:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "4"]

# Development image
FROM base as development

# Установка dev зависимостей
RUN pip install --no-cache-dir \
    pytest pytest-asyncio pytest-cov pytest-mock \
    black isort flake8 mypy \
    jupyter ipython

# Копирование кода
COPY . .

# Права доступа
RUN chown -R app:app /app
USER app

# Переменные окружения
ENV PYTHONPATH=/app
ENV ENVIRONMENT=development
ENV DEBUG=true

# Запуск в dev режиме
CMD ["uvicorn", "src.api.rest_api:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]