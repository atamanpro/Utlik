FROM python:3.12-slim

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY ./integrations /app/integrations
COPY ../common_utils /app/common_utils

CMD ["uvicorn", "integrations.main:app", "--host", "0.0.0.0", "--port", "81"]