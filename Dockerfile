FROM python:3.11-slim

# System deps for FAISS and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command — override with docker run args
CMD ["python", "scripts/run_phase1.py"]
