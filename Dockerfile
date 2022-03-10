# reduced python as base image
FROM python:3.7-slim-buster

# set a workdirectory for the app
WORKDIR /Users/heni/OneDrive/EasternUniversity/691capstone/CAPSTONE/APP/APP_KESZ/App/

# copy all the files to the container
COPY . .

# pip install dependencies from the requirements txt file
RUN pip install --no-cache-dir -r requirements.txt

# expose port 8501
EXPOSE 8501

# commands to run container
CMD ["streamlit", "run", "./app.py"]
