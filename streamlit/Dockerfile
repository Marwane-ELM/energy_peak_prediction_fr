FROM python:3.12.2-slim

WORKDIR /app

COPY streamlit/rq_streamlit.txt .

# --no-cache-dir : removes the files saved in the cache that have been used to download the libs (to free up some space)
RUN pip install --no-cache-dir -r rq_streamlit.txt

COPY streamlit/ /app/streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
