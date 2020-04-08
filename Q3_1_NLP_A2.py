import nltk 
from nltk import word_tokenize
from nltk import bigrams
from nltk import trigrams
from collections import Counter
import string 
import pandas as pd
import numpy
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


file1 = open("train_positive_reviews.txt", "r+")
file2 = open("train_negative_reviews", "r+")
words1 =  set(open("train_positive_reviews.txt").read().split())
words2 = set(open("train_negative_reviews.txt").read().split())
fin_words = words1.copy()
fin_words.update(words2)
word_list = []
count = 0
list_review = []
for ii in fin_words:
    word_list.append(ii)

a = numpy.zeros(shape=(count,len(word_list)))
df = pd.DataFrame(a,columns=word_list)
count = 0
for lines in file1:
    list_review.append(1)
    count = count + 1
    for words in lines.split():
        df.set_value(count, words, 1.0)

for lines in file2:
    list_review.append(0)
    count = count + 1
    for words in lines.split():
        df.set_value(count, words, 1.0)

df.fillna(0, inplace = True)
df.columns = [''] * len(df.columns)
X = df
y = list_review #from Q3.1
model = Sequential()
model.add(Dense(50, input_dim=len(word_list), activation='sigmoid'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=8)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
