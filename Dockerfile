FROM public.ecr.aws/docker/library/python:3.8.18-slim-bullseye
# FROM python:3.11-slim

# python:3.11.6-slim-bookworm
# public.ecr.aws/docker/library/python:3.8-slim
# python:3.8-slim

# Обновление pip
# RUN python3 -m pip install --upgrade pip

# RUN \
#     set -eux; \
#     apt-get update; \
#     DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
#     python3-pip \
#     build-essential \
#     python3-venv \
#     ffmpeg \
#     git \
#     ; \
#     rm -rf /var/lib/apt/lists/*

# alternative:
## RUN apt-get update && apt-get install -y --no-install-recommends \
###     python3-pip \
##     && rm -rf /var/lib/apt/lists/*
## RUN sed -i 's/deb.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list
# RUN pip3 install -U pip && pip3 install -U wheel && pip3 install -U setuptools==59.5.0

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && rm -r /tmp/requirements.txt

COPY . /code
WORKDIR /code

CMD ["bash"]