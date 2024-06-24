FROM python:3.12-slim

WORKDIR /app
COPY ./rag_demo/req.txt /app/req.txt
RUN pip install -r /app/req.txt

COPY ./chats_demo/yoowills/app.py /app/app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8222"]