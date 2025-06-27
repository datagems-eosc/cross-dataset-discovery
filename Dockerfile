# Use an official, slim Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy ONLY the requirements file first to leverage Docker layer caching.
# This step is only re-run if requirements.txt changes, making builds much faster.
COPY search_api/requirements.txt ./

# Install the specific dependencies for the API from its own requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Now copy your application code into the container.
# We create a search_api directory inside /app to keep the import paths clean.
COPY ./search_api /app/search_api

# Expose the port the app runs on
EXPOSE 8000

# The command to run your application in production using Gunicorn.
# Python will correctly find the `search_api.main` module.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "search_api.main:app"]