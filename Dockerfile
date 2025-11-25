# Use a slim Python base image
FROM python:3.12-slim

# Install system deps if needed (optional, can be extended later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Create and set workdir
WORKDIR /app

# Copy dependency files first (for better Docker caching)
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to install into system site-packages (no venv inside container)
RUN poetry config virtualenvs.create false \
 && poetry install --only main --no-interaction --no-ansi

# Now copy the rest of the project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command: run the API with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]