FROM python:3.12-slim

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose ports for the API
EXPOSE 8000

# Command to run the application
CMD ["python", "walden_api.py"]
