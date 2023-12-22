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
    sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)


    # Display the selected values
    st.subheader("User Input Parameter")
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}                 
    features = pd.DataFrame(data, index=[0])
    
    # Predict the target class
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]

    # Display the prediction
    species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    st.write(f"Predicted Iris Species: {species_mapping[prediction]}")

if __name__ == "__main__":
    main()
