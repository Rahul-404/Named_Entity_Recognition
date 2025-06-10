FROM python:3.9.13-slim-buster

COPY . /app

WORKDIR /app 

RUN apt-get update && apt-get install unzip && pip install --upgrade pip && pip install torch --extra-index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

CMD ["python3", "app.py"]
