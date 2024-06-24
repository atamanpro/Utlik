FROM python:3.12-slim

WORKDIR /app
COPY ./websocket_demo/req.txt /app/req.txt
RUN pip install -r /app/req.txt

COPY ./websocket_demo/web_sock.py /app/web_sock.py

CMD ["uvicorn", "web_sock:app", "--host", "0.0.0.0", "--port", "9999"]

