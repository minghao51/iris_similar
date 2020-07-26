FROM python:3.8-slim
# Set the working directory to /app
WORKDIR /app
# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

#exposing port 
EXPOSE 8050

# run unit test for data quality
RUN python test_iris_similar.py

# only run when there is no additional commands
CMD ["python", "app.py"]

# ENTRYPOINT ["python","app.py", "8000"]