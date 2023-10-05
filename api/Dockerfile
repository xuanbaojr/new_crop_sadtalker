FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \ 
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/requirements.txt

RUN python -m pip install -U --no-cache-dir --default-timeout=100 pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --no-deps -r /tmp/requirements.txt --default-timeout=100 

COPY . /app

CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8000"]