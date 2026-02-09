# PhIO Production Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY phio/ ./phio/
COPY setup.py .
COPY README.md .

# Install phio
RUN pip install -e .

# Expose port for API
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "phio.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
