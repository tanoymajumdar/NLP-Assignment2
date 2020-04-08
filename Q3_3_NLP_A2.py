import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
from gensim import models
from keras.models import Sequential
from keras.layers import Dense

model = models.KeyedVectors.load_word2vec_format("C:\GoogleNews-vectors-negative300.bin.gz", binary=True, limit = 3000)
file1 = open("train_positive_reviews.txt", "r+")
file2 = open("train_negative_reviews.txt", "r+")
fin_array = np.zeros((1,300))
count = 0
array_list = []
list_review = []
for lines in file1:
    list_review.append(1)
    for words in lines.split():    
        count = count + 1
        if words in model:
            vector = model.wv[words]
            fin_array = np.add(fin_array, vector)
        else:
            continue
    fin_array = np.true_divide(vector, count)
    array_list.append(fin_array)

for lines in file2:
    list_review.append(0)
    for words in lines.split():    

        count = count + 1
        if words in model:
            vector = model.wv[words]
            fin_array = np.add(fin_array, vector)
        else:
            continue
    fin_array = np.true_divide(vector, count)
    array_list.append(fin_array)

array_list = np.array(array_list)
X = array_list
y = list_review 
model = Sequential()
model.add(Dense(50, input_dim=300, activation='sigmoid'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=8)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
