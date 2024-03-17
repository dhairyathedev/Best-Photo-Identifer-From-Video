# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir gradio opencv-python-headless scikit-image matplotlib keras tensorflow

# Expose the port the app runs on
EXPOSE 7860

# Define the command to run your application
CMD ["python3", "model.py"]
