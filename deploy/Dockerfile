FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY deploy/ ./deploy/

# Expose ports
EXPOSE 8000 8080

# Command to run the registration server
CMD ["uvicorn", "deploy.registration_server:app", "--host", "0.0.0.0", "--port", "8000"]