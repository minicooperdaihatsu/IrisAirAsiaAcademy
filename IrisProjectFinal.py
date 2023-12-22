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
   
     # Predict the target class probabilities
    prediction_proba = model.predict_proba(input_data)[0]

 # Display the final prediction
    species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = species_mapping[prediction]
    st.write(f"Final Prediction: {predicted_species} with Probability: {max(prediction_proba):.4f}")

    # Create separate DataFrames for 'Iris Species' and 'Probability'
    species_df = pd.DataFrame({
        'Iris Species': species_mapping.values()
    })

    probability_df = pd.DataFrame({
        'Probability': prediction_proba
    })

    st.subheader("Predicted Iris Species:")
    st.dataframe(species_df)

    st.subheader("Prediction Probabilities:")
    st.dataframe(probability_df)

if __name__ == "__main__":
    main()
