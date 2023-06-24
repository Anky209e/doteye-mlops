FROM python:3.8

ENV DOCKERHOME=/code

RUN mkdir  ${DOCKERHOME}

WORKDIR ${DOCKERHOME}

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip

COPY requirements.txt ${DOCKERHOME}
COPY web.sh ${DOCKERHOME}
COPY . ${DOCKERHOME}

RUN pip install -r requirements.txt

ENTRYPOINT [ "sh","web.sh" ]