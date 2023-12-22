import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model
with open('iris_model_gnb.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
def main():
    st.title("Simple Iris Flower Prediction App")
    st.write("This app predicts the **Iris flower** type!")
    st.sidebar.header('User Input Parameters')
    
    # Use st.sidebar for sliders in the sidebar
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

    # Display the selected values as a DataFrame
    input_data = pd.DataFrame({
        'Sepal Length': [sepal_length],
        'Sepal Width': [sepal_width],
        'Petal Length': [petal_length],
        'Petal Width': [petal_width]
    })

    st.subheader("User Input parameters")
    st.dataframe(input_data)
   
    st.subheader("Class labels and their corresponding index number")
   
    st.subheader("Prediction")
        
    st.subheader("Prediction Probability")

     # Predict the target class probabilities
    prediction_proba = model.predict_proba(input_data)[0]

    # Get the predicted class
    prediction = model.predict(input_data)[0]

    # Create a DataFrame for the prediction result
    prediction_df = pd.DataFrame({
        'Iris Species': ['Setosa', 'Versicolor', 'Virginica'],
        'Probability': prediction_proba
    })

    st.subheader("Predicted Iris Species:")
    st.dataframe(prediction_df)

    # Display the final prediction
    species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    st.write(f"Final Prediction: {species_mapping[prediction]} with Probability: {max(prediction_proba):.4f}")

if __name__ == "__main__":
    main()
