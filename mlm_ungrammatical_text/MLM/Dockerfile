FROM python:3.9.14-slim-buster
WORKDIR app
COPY ./requirements.txt /app/requirements.txt
RUN apt-get update\
     && apt-get install -y\
                vim\
                git\
                curl\
                default-jre\
                gcc\
               make\
               g++\
    && pip install -r /app/requirements.txt\
    && rm -rf /var/lib/apt/lists/*
copy ./ /app
# RUN python -m nltk.downloader punkt

