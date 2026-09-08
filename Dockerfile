# use of an official Python runtime as a parent image
FROM python:3.11-slim

# set the working directory in the container
WORKDIR /app

# copy the current directory contents into the container at /app
COPY req_for_docker/requirements.txt app/requirements.txt


# install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# make port 80 available to the world outside this container
EXPOSE 8000

# copy only main.py and (any other necessary files) to the container
COPY main.py app/main.py

# Run the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]