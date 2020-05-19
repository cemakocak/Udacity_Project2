# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree - 2. Project)

## Table of Contents
1. Description
2. Dependencies
3. Executing Program
4. Additional Documents
5. Author
6. License
7. Acknowledgement

### 1. Description

Project is related with Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. Figure Eight provides disaster messages and category data. Purpose of project is to build ML Model (NLP) to correctly categorize disaster messages. Project includes three sections:
 a. ETL Pipeline to extract data from source, clean it and load into SQLite database.
 b. Machine Learning Pipeline to build and train model to classify disaster messages into categories.
 c. Design Web App to show model outputs.

### 2. Dependencies

Python 3.5+
SQLlite Database Libraqries: SQLalchemy
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
Natural Language Process Libraries: NLTK
Model Loading and Saving Library: Pickle
Web App and Visualization: Flask, Plotly

### 3.Executing Program
File sturcture of project is showed below:
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs web app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # ETL Pipeline code
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py # ML Pipeline code
|- classifier.pkl  # saved model 

- screeshots
|- Main_Page # screenshot of main web page
|- Graph 1  # screenshot of genre graph
|- Graph 2 # screenshot of category graph
|- Input  # example disaster message given as input
|- Output  # categorization of model for given message.

- README.md

You can run the following commands in the project's directory to set up the database, train model and save the model.
1. To run ETL pipeline to clean data and store the processed data in the SQLite database, execute this command: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. To run the ML pipeline that loads data from SQLite database, build model, train classifier and save the classifier as a pickle file, execute this command: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. Run the following command in the app's directory to run your web app.
    `python run.py`
4. Go to http://0.0.0.0:3001/

### 4. Additional Documents
In main directory, you can also find two jupyter notebooks:
1. ETL Pipeline Preparation: is generated before completing process_data.py code.
2. ML Pipeline Preparation: is generated before completing train_classifier.py code. This file is also used to re-train the model and tune it.

### 5. Author
Cem Akocak - Udacity Student in Data Science Nanodegree

### 6. License
Udacity 

### 7. Acknowledgement
Udacity for providing great online lessons in Data Science Nanodegree Program
Figure Eight for providing dataset to train model