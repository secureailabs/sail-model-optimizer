# Use a base image with Python and dependencies pre-installed
FROM python:3.8.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code to the container
COPY server/main.py .

# Copy the data and results directories to the container (DEMO)
COPY script/flf/data data
COPY script/flf/results results

# Copy the model and optimizer directories to the container and install
COPY sail_model_optimizer sail_model_optimizer
COPY setup.py .
COPY setup.cfg .
COPY release.py .
COPY README.md .

# Expose the port on which the FastAPI server will listen
EXPOSE 8000

# Start the FastAPI server
CMD pip install -e . && uvicorn main:app --host 0.0.0.0 --port 8000
