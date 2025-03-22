# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies needed for psycopg2 and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000 for the app (Render dynamically assigns $PORT, so EXPOSE is optional)
EXPOSE 8000

# Set the command to run the app using Uvicorn with dynamic port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
