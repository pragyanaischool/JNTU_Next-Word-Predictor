import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Ensure compatibility for deprecated TensorFlow functions
if hasattr(tf.compat.v1, "reset_default_graph"):
    tf.compat.v1.reset_default_graph()

# Load the LSTM Model
model = load_model('nextwordlstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.set_page_config(page_title="Next Word Predictor", page_icon="✨", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f9;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .title {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 20px;
        }
        .subtitle {
            text-align: center;
            color: #6c757d;
            margin-bottom: 40px;
        }
        .input-box {
            margin: 0 auto;
            width: 50%;
        }
        .prediction {
            text-align: center;
            font-size: 24px;
            color: #4CAF50;
        }
        .footer {
            text-align: center;
            font-size: 12px;
            color: #999;
            margin-top: 50px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown("<h1 class='title'>Next Word Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='subtitle'>Using LSTM Neural Networks for Predictive Text</h4>", unsafe_allow_html=True)
st.image("PragyanAI_Transperent_github.png")
# User input
st.markdown("<div class='input-box'><h5>Enter a sequence of words:</h5></div>", unsafe_allow_html=True)
input_text = st.text_input("", placeholder="Type your sentence here...")

# Prediction button and output
if st.button("Predict Next Word"):
    if input_text.strip() == "":
        st.error("Input cannot be empty. Please enter some text.")
    else:
        max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.markdown(f"<div class='prediction'>Predicted Next Word: <strong>{next_word}</strong></div>", unsafe_allow_html=True)
        else:
            st.warning("Prediction failed. Try a different input.")

# Footer
st.markdown(
    """
    <div class='footer'>
        Created with ❤️ using Streamlit | © 2025
    </div>
    """,
    unsafe_allow_html=True
)
