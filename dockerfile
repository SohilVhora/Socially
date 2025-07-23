# Stage 1: Build the React frontend
FROM node:18-alpine AS frontend
WORKDIR /app
COPY fe/package*.json ./
RUN npm install
COPY fe/ .
RUN npm run build

# Stage 2: Build the Python backend
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY be/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY be/ .
COPY --from=frontend /app/dist ./fe/dist

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
