import pickle 
import numpy as np
import streamlit as st 

from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.sequence import pad_sequences 

# load our model 
model = load_model(r'pickle_files/next_word_lstm.h5')

# load tokenizer 
with open(r'pickle_files/tokenizer.pkl','rb') as file_obj:
    tokenizer = pickle.load(file_obj)

# prediction function
def predict_next_word(model, tokenizer, text, maxSeqLen):
    token_list = tokenizer.texts_to_sequences([text])[0] # converts text to vocabulary indexes
    if len(token_list) >= maxSeqLen:
        token_list = token_list[-(maxSeqLen-1):] # taking last maxSeqLen-1 words
    else:
        token_list = pad_sequences([token_list],maxlen=maxSeqLen-1,padding='pre')
    
    y_pred = model.predict(token_list,verbose=0) # 2d arr of vocabulary size
    predicted_word_index = np.argmax(y_pred,axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    
    return None


st.title('Next Word Prediction with LSTM')

input_text = st.text_input('Enter the sequences of words here','i am going to')

if st.button("Predict next word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'The next word is: {next_word}')


