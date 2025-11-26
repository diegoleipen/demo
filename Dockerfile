# Use a slim Python base image
FROM python:3.13-slim

# Optional: system deps if you ever need them
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Create and set workdir
WORKDIR /app

# Copy the project into the image
COPY . .

# Install the project (and its runtime dependencies) using PEP 517 backend (poetry-core)
# This reads your pyproject.toml and installs [project].dependencies
RUN pip install --no-cache-dir .

# Expose FastAPI port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]