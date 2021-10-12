FROM nvidia/cuda:11.0-base
FROM python:3.8.10
USER root

WORKDIR /workspace
COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8088

CMD ['flask', 'run', '--host=0.0.0.0 -p 8088']