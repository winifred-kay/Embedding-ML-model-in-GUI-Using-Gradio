# Embedding-ML-model-in-GUI-Using-Gradio

## Introduction
Building models that solves real life problems is great but not so useful if it cannot be wrapped up nicely into a friendly interface that could benefit non technical people. This is where gradio becomes a go to tool for data experts.
##### What is Gradio?
Gradio is an open-source Python package that allows you to quickly create easy-to-use, customizable UI components for your ML model, any API, or even an arbitrary Python function using a few lines of code. 

This project seeks to use streamlit to create a friendly interface for a sales prediction model created to predict sales based on a dataset.

## Required Installations
1. To run this project you should have python 3 and above installed
2. Install streamlit using pip
3. Have anaconda or visual studio code
4. Not a requirement but advisable to create a virtual environment to run.
5. Replace your image location in the app.py file
6. Copy and paste the lie below in your terminal to run this project.
    streamlit run "your file location/app.py"
  
## Project Flow
1. Import all required libraries
2. Load machine learning model
3. Create an interface to take inputs
4. Extract other individual components from model ( encoders,scalers, imputers)
5. Create a dataframe for expected inputs
6. Split categorical and numerical inputs
7. Apply enconders on categorical colums and scale numerical inputs
8. Join back categorical and numerical inputs
9. Pass manipulated inputs through the model
10. Make preictions
