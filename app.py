import streamlit as st
import pickle
import numpy as np
import warnings

# Ignore the warning
warnings.filterwarnings("ignore", category=UserWarning)


with open('x_encoder.pkl', 'rb') as model_file:
    x_encoder = pickle.load(model_file)

with open('y_encoder.pkl', 'rb') as model_file:
    y_encoder = pickle.load(model_file)

# Load the trained model
with open('Disorder_Predictor_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit UI
def main():
    st.title("Mental Health Disorder Prediction")

    # Get user input for 24 features
    features = ['Are you feeling nervous?', 'Are you panic?', 'Are you breathing rapidly?', 'Is your body sweating frequently?',
       'Are you facing trouble in concentration?', 'Are you having trouble in sleeping?',
       'Are you having trouble with work?', 'Are you feeling hopelessness?', 'Are you be angery frequenty?', 'Are you over reacting on silly things?',
       'Are there frequently change in eating?', 'Are you have suicidal thought?', 'Are you feeling tired alltime?', 'Do you have any close friend?',
       'Do you have social media addiction?', 'Are you facing weight gain unusually?', 'Are you have material possessions?',
       'Are you introvert?', 'Do you struggle with popping up stressful memory?', 'Do you having nightmares?',
       'Do you avoids people or activities?', 'Are you feeling negative?',
       'Are you facing trouble concentrating in work?', 'Are you blamming yourself?']
    user_inputs = []
    for i in range(24):
        # feature_name = f"Feature {i + 1}"
        feature_name = features[i]
        # user_input = st.text_input(feature_name, "yes")
        # user_input = st.selectbox(feature_name, options=["no", "yes"])
        user_input = st.radio(feature_name, options=["no", "yes"])
        user_inputs.append(user_input)

    # Convert user inputs using the loaded LabelEncoders
    encoded_inputs = []
    for i, user_input in enumerate(user_inputs):
        try:
            encoded_value = x_encoder.transform([user_input])[0]
            encoded_inputs.append(encoded_value)
        except Exception as e:
            st.write(f"Error processing Feature {i + 1}: {e}")
            encoded_inputs = None
            break

    # Make predictions when valid inputs are provided
    if encoded_inputs is not None and len(encoded_inputs) == 24:
        prediction = model.predict([encoded_inputs])
        predicted = y_encoder.inverse_transform(prediction)[0]
        st.write("Prediction:", predicted)
    else:
        st.write("Invalid input. Please provide valid input for all features.")

# Run the app
if __name__ == "__main__":
    main()
