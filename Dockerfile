# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container
COPY chatbot.py .

# Install NLTK
RUN pip install nltk

# Download NLTK data
RUN python -m nltk.downloader punkt

# Run the chatbot script when the container starts
CMD ["python", "chatbot.py"]
