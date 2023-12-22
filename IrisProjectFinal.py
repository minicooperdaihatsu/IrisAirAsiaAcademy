import streamlit as st
import pickle

# Load the pre-trained model
with open('iris_model_gnb.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
def main():
    st.title("Simple Iris Flower Prediction App")
    st.write("This app predicts the **Iris flower** type!")
    
    # Use st.sidebar for sliders in the sidebar
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}                 
    features = pd.DataFrame(data, index=[0])

    # Display the selected values
    st.subheader("User Input Parameter")
    st.write(f"Sepal Length: {sepal_length}")
    st.write(f"Sepal Width: {sepal_width}")
    st.write(f"Petal Length: {petal_length}")
    st.write(f"Petal Width: {petal_width}")
    
    # Predict the target class
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]

    # Display the prediction
    species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    st.write(f"Predicted Iris Species: {species_mapping[prediction]}")

if __name__ == "__main__":
    main()
