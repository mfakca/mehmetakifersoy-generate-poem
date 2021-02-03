#Importing the required packages
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())


'''

# reading book
with open ("mehmetAkifErsoyTemiz.txt","r",encoding="utf-16") as siir:
    seq = siir.read().split("\n")


seq = [i for i in seq if i != ""]



seq = list(map(lambda x: x.strip(),seq))




# get lower
lower = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRŞSTUÜVYZ", "abcçdefgğhıijklmnoöprşstuüvyz")
seq = list(map(lambda x: x.translate(lower), seq))


#Remove punct

import re

seq = list(map(lambda x: re.sub(r'[^\w\s]','',x),seq))


# Tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(seq)
total_no_words = len(tokenizer.word_index) + 1


print(tokenizer.word_index)



#input sequence
input_poem_sequences = []
for line in seq:
    tokenlist = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokenlist)):
        ngram_sequence = tokenlist[:i+1]
        input_poem_sequences.append(ngram_sequence)

print(input_poem_sequences[:20])


# Input Sequence Padding

max_input_sequence_len = max([len(x) for x in input_poem_sequences])
input_poem_sequences = np.array(pad_sequences(input_poem_sequences, maxlen=max_input_sequence_len, padding='pre'))

print(max_input_sequence_len)
print(input_poem_sequences)



import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

model= Sequential()
model.add(Embedding(total_no_words, 100,input_length=max_input_sequence_len-1)) 
model.add(Bidirectional(LSTM(150, return_sequences = True))) 
model.add(Dropout(0.2))
model.add(LSTM(100)) 
model.add(Dense(total_no_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.001))) 
model.add(Dense(total_no_words, activation='softmax')) 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) 
model.summary()


model_predictors, output_label = input_poem_sequences[:,:-1],input_poem_sequences[:,-1]
output_label = ku.to_categorical(output_label, num_classes=total_no_words)
print(model_predictors)
#In this step we are now fitting the model.
history = model.fit(model_predictors, output_label ,epochs=200, verbose =1)

model.save("deneme_gpu_200_epoch.h5")




def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()
    
print(generate_text("gemiler geldi oturdu",10,model,max_input_sequence_len))

'''