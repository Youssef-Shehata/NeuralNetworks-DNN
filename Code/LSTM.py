import sys 
import os
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import keras
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath("/home/joe/School/Neural/NeuralNetworks-DNN/")))


def label_decoder(label):
    if label == "0":
        return -1
    elif label == "1":
        return 0
    else:
        return 1




def mylstm(num_unique_words, max_sequence_length, train_padded, train_labels, val_padded, val_labels , test_sentences):
  
    # # print(train_padded.shape)
    # # print(test_sentences.shape)

    # # print(train_padded[0:3])
    # # print(test_sentences[0:3])
    
    
    # # # Creating model
    model = Sequential()



    model.add(Embedding(num_unique_words + 1 , 150, input_length=max_sequence_length, mask_zero=True))

    model.add(LSTM(128, dropout=0.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    loss = keras.losses.sparse_categorical_crossentropy
    optim = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy'])
    model.fit(train_padded, train_labels, epochs=3, validation_data=(val_padded, val_labels), verbose=2)


    score = model.evaluate(val_padded, val_labels, verbose=2)
    print(f"Test Accuracy:", score[1])

    predictions = model.predict(test_sentences)
    predictions = np.argmax(predictions, axis=1) - 1


    for i in range(len(train_labels)):
        train_labels[i] = label_decoder(train_labels[i])
    print("Actual labels : ", train_labels[10:20])    
    print("Predicted labels : ", predictions[10:20])



    data = pd.read_csv("/home/joe/School/Neural/NeuralNetworks-DNN/Data/test _no_label.csv")
    submimssion= pd.DataFrame()
    submimssion["ID"] = data['ID']
    submimssion["rating"] = predictions
    submimssion.to_csv("LSTMsubmission.csv", index=False)




# Example usage:
# mylstm(num_unique_words, max_sequence_length, train_padded, train_labels, val_padded, val_labels)
