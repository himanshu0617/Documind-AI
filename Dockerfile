# Use official lightweight Python 3.12 image
FROM python:3.12-slim

# Set environment variables for non-interactive installs and unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install minimal OS dependencies for PyMuPDF and C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root system user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy application source code
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000 8501

# Health check against FastAPI endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command launches FastAPI backend
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
