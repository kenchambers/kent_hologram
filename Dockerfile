# Multi-stage Dockerfile for Kent Hologram Dashboard
# Stage 1: Build frontend
# Stage 2: Python runtime with FastAPI serving static files

# ============================================
# Stage 1: Build Next.js frontend
# ============================================
FROM node:22-alpine AS frontend-builder

WORKDIR /frontend

# Copy package files
COPY web/frontend/package.json web/frontend/package-lock.json* ./

# Install dependencies
RUN npm ci --prefer-offline

# Copy frontend source
COPY web/frontend/ ./

# Build Next.js app (static export)
RUN npm run build

# ============================================
# Stage 2: Python runtime
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster pip
RUN pip install uv

# Copy Python project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY web/backend/ ./web/backend/

# Install Python dependencies
RUN uv pip install --system -e . && \
    uv pip install --system fastapi uvicorn[standard] python-multipart

# Copy built frontend (static export)
COPY --from=frontend-builder /frontend/out ./frontend/

# Create data directory
RUN mkdir -p /app/data

# Environment
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
CMD ["uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
