FROM python:2

LABEL maintainer="s.verhoeven@esciencecenter.nl"

RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/

WORKDIR /code/ms2ldaviz

ENV DJANGO_SETTINGS_MODULE=ms2ldaviz.settings_docker

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

EXPOSE 8000
