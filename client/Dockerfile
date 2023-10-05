FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \ 
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U --no-cache-dir --default-timeout=100 pip && pip install --no-cache-dir --upgrade -r /app/requirements.txt --default-timeout=100

CMD ["python", "app_gradio.py"]