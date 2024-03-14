# Use an official Python runtime as a parent image
FROM python:3.7.0


# Set the working directory in the container
WORKDIR /usr/src/app

# Copy all the files from here into the container
COPY . /app/

# Install TensorFlow 1.15.5
RUN pip install --upgrade pip
RUN pip install tensorflow==1.14.0
RUN  pip install pandas


# CMD specifies the default command to run when the container starts
# CMD ["python", "your_script.py"]
