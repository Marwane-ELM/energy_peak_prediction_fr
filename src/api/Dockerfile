FROM python:3.12.3

WORKDIR /app

COPY src/api/ .

# --no-cache-dir : removes the files saved in the cache that have been used to download the libs (to free up some space)
RUN pip install --no-cache-dir -r rq_api.txt

EXPOSE 8000

CMD ["fastapi", "run", "get_preds.py", "--host", "0.0.0.0", "--port", "8000"]