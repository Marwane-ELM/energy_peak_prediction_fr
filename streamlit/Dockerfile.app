FROM python:3.12.3

WORKDIR /app

COPY requirements.txt .

COPY streamlit/ .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["fastapi", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8000"]