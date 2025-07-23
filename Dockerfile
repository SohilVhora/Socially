# Stage 1: Build the React frontend
FROM node:18-alpine AS frontend
WORKDIR /app
COPY fe/package*.json ./
RUN npm install
COPY fe/ .
RUN npm run build

# Stage 2: Create the Python production environment
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies (like for the 'soundfile' library)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY be/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code and the built frontend from the previous stage
COPY be/ .
COPY --from=frontend /app/dist ./fe/dist

# Expose the port the backend runs on
EXPOSE 8000

# Run the backend server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
