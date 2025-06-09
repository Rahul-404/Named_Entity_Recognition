FROM python:3.9.13-slim-buster

COPY . /named_entity_recognition

WORKDIR /named_entity_recognition 

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]