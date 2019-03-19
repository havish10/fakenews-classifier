FROM python:3

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD [ "python", "./index.py" ]
EXPOSE 5000