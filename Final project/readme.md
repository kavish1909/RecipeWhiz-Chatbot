************Steps for running the project*****************

1) Install anaconda/any ide along with python in your machine.

Link for ananconda installation: https://docs.anaconda.com/free/anaconda/install/windows/
Link for python installation: https://www.python.org/downloads/windows/
These links are specifically for users with windows os

Use below commands in anaconda prompt
-pip install nltk 
-pip install flask (To install flask framework)
-pip install --upgrade threadpoolctl (controlling the number of threads used by native libraries)
-pip install fsspec (unified interface for working with different file systems and remote storage systems)
Rest of the dependencies are installed automatically via the scripts.
 

2) Once step 1 is complete
Download recipe dataset from https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews
The dataset should be in .csv format and should be placed in root folder along with .py scripts

3) Run the script data_clean.py(This script performs data cleaning)
-Open anaconda prompt and locate directory where project is present. Use cd 'Project path'
-Use command 'python data_clean.py' (This will created a cleaned.csv file as an output)
-This file will be used by classification, clustering and flask scripts.

4) Run the script classification.py
-Locate the project in anaconda prompt using cd command
-Use command 'python classification.py'(This will run the logistic regression model)
-This will create model files (vectorizer.pkl, logistic_regression_model.pkl)

5) Run the script clustering.py
-Locate the project in anaconda prompt using cd command
-Use command 'python clustering.py'(This will run the MiniBatchKMeans model)
-This will create model files (clustering.pkl)

6) Run the flask app (app.py)
-Locate the project in anaconda prompt using cd command
-Use command 'python app.py'(This will run the flask web app and will give a link to open chatbot on browser)
-Now you can interact with chatbot.