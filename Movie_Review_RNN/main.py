import numpy as np 
import streamlit as st
import tensorflow as tf 

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

model = load_model('./pickle_files/simple_rnn_imdb.h5')

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items() }


# Function to Decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'***') for i in encoded_review])

# Function to preprocess user data 
def preprocess(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=1000)
    return padded_review

# Prediction Function:
def predict_sentiment(review):

    data = preprocess(review)
    prediction = model.predict(data)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


# st.title('IMDB Movie Review Sentiment Analysis')
# st.write('Enter the movie review')

# # user input
# user_input = st.text_area('Movie Area')

# if st.button('Classify'):
#     sentiment,score = predict_sentiment(user_input)

#     st.write(f"The review was {sentiment} with probability of review being positive as {score}")
# else:
#     st.write('Please enter the movie review')

import streamlit as st

# Custom CSS for background and text styling
st.markdown(
    """
    <style>
    /* Background styling */
    body {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
    }
    /* Center the title */
    .main-title {
        text-align: center;
        font-size: 40px;
        color: #4CAF50;
        font-weight: bold;
        margin-bottom: 20px;
    }
    /* Input box styling */
    textarea {
        font-size: 16px;
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 10px;
        background-color: #fff;
    }
    /* Button styling */
    button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }
    button:hover {
        background-color: #45a049;
    }
    /* Sentiment result styling */
    .sentiment-result {
        font-size: 20px;
        color: white;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Application title
st.markdown('<div class="main-title">IMDB Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)

# Instruction
st.write('Enter the movie review below:')

# User input
user_input = st.text_area('Movie Area', placeholder="Type your movie review here...")

# Sentiment classification button
if st.button('Classify'):
    sentiment, score = predict_sentiment(user_input)

    # Display the result
    st.markdown(
        f'<div class="sentiment-result">The review was <strong>{sentiment}</strong> with a probability of the review being positive as <strong>{score:.2f}</strong>.</div>',
        unsafe_allow_html=True,
    )
else:
    st.write('Please enter the movie review.')


# indices[0,999] = 53892 is not in [0, 10000)
#          [[{{node sequential/embedding/embedding_lookup}}]] [Op:__inference_predict_function_3250]

# words = text.lower().split()
#     encoded_review = [word_index.get(word,2)+3 for word in words]
#     padded_review = sequence.pad_sequences([encoded_review],maxlen=1000)

# max_len = 1000 
# model = Sequential()
# model.add(Embedding(max_features,300,input_length=max_len))
# model.add(SimpleRNN(128,activation=LeakyReLU(alpha=0.1)))
# model.add(Dense(1,activation="sigmoid"))