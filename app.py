import streamlit as st
import numpy as np
import pickle

# Set the background image and text color
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://media.istockphoto.com/id/1459216440/photo/gym-room-fitness-center-interior-with-equipment-and-machines.jpg?s=2048x2048&w=is&k=20&c=xGkQ_T8MOUW60Fxl0pbSd-61T_CazlgGjFHIbM2Ce6Q=");
    background-size: 100vw 100vh;
    background-position: center;  
    background-repeat: no-repeat;
    color: #FFFFFF; /* Set text color to blue */
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)


# Load the trained XGBoost model
with open("xgboost_calories_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define a function to make predictions
def predict_calories(features):
    prediction = model.predict(features)
    return prediction[0]

# Create the Streamlit web app
def main():
    # Set the title
    st.title("Calories Burnt Prediction App")

    # Add input elements for user to enter values
    st.header("Enter Exercise Details:")
    age = st.number_input("Age", min_value=1, max_value=150, step=1)
    gender = st.radio("Gender", options=["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=1.0, max_value=300.0, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=500.0, step=0.1)
    duration = st.number_input("Duration (minutes)", min_value=1, max_value=1000, step=1)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=1, max_value=300, step=1)
    body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1)

    # Map gender to numerical value
    gender_mapping = {"Male": 0, "Female": 1}
    gender_numeric = gender_mapping[gender]

    # Make prediction when the "Predict" button is clicked
    if st.button("Predict Calories Burnt"):
        # Prepare input features as a numpy array
        features = np.array([[gender_numeric, age, height, weight, duration, heart_rate, body_temp]])

        # Predict calories burnt
        prediction = predict_calories(features)
        st.success(f"Predicted calories burnt: {prediction:.2f} calories")

if __name__ == "__main__":
    main()
