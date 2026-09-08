FROM python:3.12.3

WORKDIR /app

COPY src/api/ .

RUN pip install --no-cache-dir -r rq_api.txt

EXPOSE 8000

CMD ["fastapi", "run", "get_preds.py", "--host", "0.0.0.0", "--port", "8000"]