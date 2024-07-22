
# Sales Prediction Using Python - CODSOFT INTERNSHIP TASK

Sales prediction involves forecasting the amount of a product that customers will purchase, taking into account various factors such as advertising expenditure, target audience segmentation, and advertising platform selection.

In businesses that offer products or services, the role of a Data Scientist is crucial for predicting future sales. They utilize machine learning techniques in Python to analyze and interpret data, allowing them to make informed decisions regarding advertising costs. By leveraging these predictions, businesses can optimize their advertising strategies and maximize sales potential. Let's embark on the journey of sales prediction using machine learning in Python.

## Description

The project consists of two main files:

1) model.py: This file contains the code to train and run the machine learning model efficiently.

2) app.py: This file sets up a Streamlit web application for an interactive and user-friendly interface to input sales-related details and predict sales.
The dataset used for this project is advertising.csv, which contains various features related to advertising and sales.

You can run either model.py for a command-line interface or app.py for a graphical user interface. app.py provides a better user experience by using Streamlit to interact with the model.
## Acknowledgements

We would like to thank the following resources and individuals for their contributions and support:

Streamlit: For offering an easy-to-use framework for deploying machine learning models.

Scikit-learn: For providing powerful machine learning tools and libraries.

Pandas: For data manipulation and preprocessing.

NumPy: For numerical operations.


## Demo

https://drive.google.com/file/d/1a7_FTVf5st-P69JKZuOz1x2lsseFvXYK/view?usp=drive_link

You can see a live demo of the application by running the app.py file. The Streamlit app allows you to input sales-related details and get a predicted sales amount based on the trained model.
## Features

Data Loading and Preprocessing: The model can load and preprocess data from the advertising.csv file.

Model Training: Utilizes a machine learning algorithm to train the model on sales data.

Interactive User Input: Through the Streamlit app, users can input sales-related details and receive a predicted sales amount.

Feature Importance: Displays the importance of each feature in predicting sales.

## Technologies Used

Python: The programming language used to implement the model and the Streamlit app.

Pandas: For data manipulation and preprocessing.

NumPy: For numerical operations.

Scikit-learn: For building and training the machine learning model.

Streamlit: For creating the interactive web application.
## Installation

to get started with this project, follow these steps:

1) Clone the repository:

git clone https://github.com/SiddharthWayne/SALES-PREDICTION-USING-PYTHON---CODSOFT-INTERNSHIP-TASK.git

cd sales-prediction

2) Install the required packages:

pip install -r requirements.txt

Ensure that requirements.txt includes the necessary dependencies like pandas, numpy, scikit-learn, and streamlit.

3) Download the dataset:

Place the advertising.csv file in the project directory. Make sure the path in model.py and app.py is correctly set to this file.
## Usage/Examples 

1) Running the Model (model.py) :

To train and run the model using the command line, execute the following:

python model.py

This will train the model and allow you to input sales-related details via the command line interface to get a predicted sales amount.

2) Running the Streamlit App (app.py):

To run the Streamlit app for an interactive experience, execute the following:
streamlit run app.py

This will start the Streamlit server, and you can open your web browser to the provided local URL to use the app.

Example:

Once the Streamlit app is running, you can input details such as:

TV Advertising Budget: $230.1

Radio Advertising Budget: $37.8

Newspaper Advertising Budget: $69.2

Click the "Predict Sales" button to get the predicted sales 
amount.