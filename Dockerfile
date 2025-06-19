

# Use Java 20 instead of Java 17
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR .

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python application
COPY app/src/main/python .

# Copy models directory
COPY app/src/main/models ./models

# Add python path to include the src directory
ENV PYTHONPATH="${PYTHONPATH}:/app/src/main/python"

# Expose port 8080 for the application
EXPOSE 8080

# Run the Python application
CMD ["python", "app.py"]
