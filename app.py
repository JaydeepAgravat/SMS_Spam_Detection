import pickle
import os
from transformers import *
import streamlit as st

# Load pre-trained model
with open('text_classification_pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)


# Function to predict spam or ham
def predict_spam_or_ham(text):
    prediction = loaded_pipeline.predict([text])[0]
    return 'Spam' if prediction == 1 else 'Ham'


# Main function to create UI
def main():
    # Set page title and icon
    st.set_page_config(
        page_title='Spam SMS Detection',
        page_icon=":envelope_with_arrow:",
        layout='wide'
    )

    # Set app title and description
    st.title('Spam SMS Detection')
    st.write("Welcome to the Spam SMS Detection App! Enter a message to classify whether it's spam or ham.")

    # Text input area
    text_to_classify = st.text_area("Enter SMS to Detect:", height=150)

    # Classify button
    if st.button("Detect", key='classify_button'):
        # Display prediction
        prediction = predict_spam_or_ham(text_to_classify)
        if prediction=='Ham':
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"Prediction: {prediction}")

    # Add horizontal rule for separation
    st.markdown("---")

    # About section
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a machine learning model to classify SMS messages as either spam or ham. "
    )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Developed by [Jaydeep Agravat](https://github.com/JaydeepAgravat)"
    )


if __name__ == "__main__":
    main()