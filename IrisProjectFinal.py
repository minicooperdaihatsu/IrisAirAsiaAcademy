import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
from sklearn.naive_bayes import GaussianNB

# Load the pre-trained model
with open('iris_model_gnb.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
def main():
    st.title("Simple Iris Flower Prediction App")
    st.write("This app predicts the **Iris flower** type!")

st.sidebar.header('User Input Parameters')

    # Feature sliders for input values
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

    # Predict the target class
   data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}  
    features = model.predict(data, index=[0])
    return features
    
    # Display the prediction
    species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    st.write(f"Predicted Iris Species: {species_mapping[prediction]}")

if __name__ == "__main__":
    main()
