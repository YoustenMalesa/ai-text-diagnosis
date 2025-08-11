FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY data ./data
COPY src ./src
RUN mkdir -p models
# Training will occur on container startup if model missing (see start_server.py)
EXPOSE 8000
CMD ["python", "-m", "src.start_server"]
