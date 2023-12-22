import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model
with open('iris_model_gnb.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
def main():
    st.title("Iris Flower Prediction App (Gaussian Naive Bayes)")

    # Use st.sidebar for sliders in the sidebar
    sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)

    # Display the selected values as a DataFrame
    input_data = pd.DataFrame({
        'Sepal Length': [sepal_length],
        'Sepal Width': [sepal_width],
        'Petal Length': [petal_length],
        'Petal Width': [petal_width]
    })

    st.subheader("Selected Input Values:")
    st.dataframe(input_data)

    # Predict the target class probabilities
    prediction_proba = model.predict_proba(input_data)[0]

    # Display the prediction as an array
    st.subheader("Predicted Probabilities:")
    st.write(f"Setosa: {prediction_proba[0]:.4f}")
    st.write(f"Versicolor: {prediction_proba[1]:.4f}")
    st.write(f"Virginica: {prediction_proba[2]:.4f}")

    # Get the predicted class
    prediction = model.predict(input_data)[0]

    # Display the predicted class
    species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    st.write(f"Predicted Iris Species: {species_mapping[prediction]}")

if __name__ == "__main__":
    main()
