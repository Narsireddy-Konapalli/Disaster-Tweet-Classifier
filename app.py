import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

model_name = 'Saved Model' 

loaded_model = TFBertForSequenceClassification.from_pretrained(model_name)
loaded_tokenizer = BertTokenizer.from_pretrained(model_name)

st.title("Disaster Tweet Classifier")
st.write("Enter a tweet and classify it as **Ham** (not disaster-related) or **Spam** (disaster-related).")

def predict_tweet(tweet_text):
    inputs = loaded_tokenizer(tweet_text, return_tensors='tf', padding=True, truncation=True, max_length=512)
    
    
    with tf.device('/CPU:0'): 
        predictions = loaded_model(inputs)
    
    predicted_class = np.argmax(predictions.logits, axis=-1)[0]
    return predicted_class

user_input = st.text_area("Enter the tweet text here:", "")


if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = predict_tweet(user_input)
        label = "Spam (Disaster-related)" if prediction == 1 else "Ham (Not disaster-related)"
        st.success(f"Prediction: **{label}**")
