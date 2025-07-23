# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all backend files
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port for the FastAPI server
EXPOSE 8000

# Run the server using uvicorn on server.py
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]