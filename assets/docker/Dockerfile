FROM python:3.6.5-jessie

RUN apt-get update && apt-get install -y \
        mecab \
        libmecab-dev \
        mecab-ipadic \
        mecab-ipadic-utf8 \
        nkf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
