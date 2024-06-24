FROM python:3.12-slim

WORKDIR /app
COPY ./chats_demo/req.txt /app/req.txt
RUN pip install -r /app/req.txt

COPY ./chats_demo/relaxsan/app.py /app/app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8222"]