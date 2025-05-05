import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

# Load the saved model and tokenizer
model_name = 'Saved Model'  # Replace with the actual path to your saved model

# Load the BERT tokenizer and model from the saved directory
loaded_model = TFBertForSequenceClassification.from_pretrained(model_name)
loaded_tokenizer = BertTokenizer.from_pretrained(model_name)

# Title
st.title("Disaster Tweet Classifier")
st.write("Enter a tweet and classify it as **Ham** (not disaster-related) or **Spam** (disaster-related).")

# Function to preprocess input text and get predictions
def predict_tweet(tweet_text):
    # Tokenize input text
    inputs = loaded_tokenizer(tweet_text, return_tensors='tf', padding=True, truncation=True, max_length=512)
    
    # Get model predictions
    with tf.device('/CPU:0'):  # Ensure to use CPU to avoid GPU memory issues
        predictions = loaded_model(inputs)
    
    # Get the predicted class (0: Ham, 1: Spam)
    predicted_class = np.argmax(predictions.logits, axis=-1)[0]
    
    return predicted_class

# Text input for user
user_input = st.text_area("Enter the tweet text here:", "")

# Predict button
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = predict_tweet(user_input)
        label = "Spam (Disaster-related)" if prediction == 1 else "Ham (Not disaster-related)"
        st.success(f"Prediction: **{label}**")
