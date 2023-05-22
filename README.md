# Breast_Cancer_Prediction_App
Machine Learning App to Predict Cancer

## Overview

The Breast Cancer Diagnosis app is an application that employs machine learning to aid healthcare providers in identifying breast cancer. It utilizes a range of measurements to determine whether a breast mass is benign or malignant. The app generates a radar chart to visually present the input data and offers the predicted diagnosis along with the probability of it being benign or malignant. Users have the option to manually input the measurements or link the app to a cytology lab to directly acquire the data from a machine. It is important to note that the app does not include the functionality to directly connect to the laboratory machine; this connection needs to be established separately.

The application was created as a machine learning practice using the public dataset called  [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). It should be noted that this dataset may not be trustworthy since the project was solely developed for educational purposes in the field of machine learning and not intended for professional use.

A live version of the application can be found on [Streamlit Community Cloud](https://alejandro-ao-streamlit-cancer-predict-appmain-uitjy1.streamlit.app/). 

## Installation

In order to use the Cell Image Analyzer on your local machine, it is necessary to have Python 3.6 or a newer version installed. Subsequently, you can install the necessary packages by executing the following command:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies, including Streamlit, OpenCV, and scikit-image.

## Usage
To start the app, simply run the following command:

```bash
streamlit run app.py
```


### To run the application
```
stremlit run app/main.py
```