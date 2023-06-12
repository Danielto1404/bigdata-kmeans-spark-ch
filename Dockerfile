FROM jupyter/pyspark-notebook

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD requirements.txt /app/

RUN pip install -r requirements.txt

ADD . /app