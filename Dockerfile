FROM public.ecr.aws/docker/library/python:3.11.6-slim-bullseye

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements_indexes.txt && rm -r /tmp/requirements.txt

COPY . /code
WORKDIR /code

CMD ["bash"]