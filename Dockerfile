FROM python:3.11-slim

WORKDIR /workspace

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default to bash so users can run scripts interactively
CMD ["/bin/bash"]
