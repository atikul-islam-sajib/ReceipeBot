FROM python:3.12-alpine

WORKDIR /app/

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "src/receipe_bot.py"]
