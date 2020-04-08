from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
file1 = open('C:\pos_test.txt', 'r+')
file2 = open ('C:\\neg_test.txt', 'r+')
sentence_list = [] 
list_review = []
words1 =  set(open("train_positive_reviews.txt").read().split())
words2 = set(open("train_negative_reviews.txt").read().split())
fin_words = words1.copy()
fin_words.update(words2)
word_list = []

for ii in fin_words:
    word_list.append(ii)
for lines in file1:
    sentence_list.append(lines)
    list_review.append(1)
for lines in file2:
    sentence_list.append(lines)
    list_review.append(0)

vectorizer = TfidfVectorizer()

m = vectorizer.fit_transform(sentence_list)

X = m
y = list_review 
model = Sequential()
model.add(Dense(50, input_dim=len(word_list), activation='sigmoid'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=8)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
