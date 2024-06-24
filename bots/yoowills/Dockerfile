FROM python:3.12-slim

WORKDIR /app
COPY ./bots/yoowills/req.txt /app/req.txt
RUN pip install -r /app/req.txt

COPY ./bots/yoowills/app.py /app/app.py

CMD ["python", "app.py"]