import spacy
from collections import Counter
import re

str = open('parsing_dataset.txt', 'r').read()
file = open('Q1.txt', 'w+')
sentence_count = 0
entity_set = set() 
verb_count = 0
nlp = spacy.load("en_core_web_sm")
doc = nlp(str)
file.writelines("POS tags:")
for token in doc:
    file.writelines(token.tag_)
    continue
file.writelines("\n")
file.writelines("Entities:")
for entity in doc.ents:
    entity_set.add(entity.label_)
file.writelines(entity_set)
file.writelines("\n")
prep_count = set()
prep_list = []
for sent in doc.sents:    
    sentence_count = sentence_count + 1 

for token in doc:
    if token.pos_ == "VERB":
        verb_count = verb_count + 1
    if token.tag_ == "IN":
        prep_count.add(token)
        prep_list.append(token)
var = verb_count/sentence_count
file.writelines("Preposition count:")
file.writelines(len(prep_count))
file.writelines('\n')
file.writelines("Most common prepositions:")
file.writelines(Counter(words).most_common(3))
file.writelines('\n')


file.writelines("Sentence count:")
file.writelines(str(sentence_count))
file.writelines('\n')
file.writelines("Average verb count:")
file.writelines(str(var))
file.writelines('\n')
file.writelines("entity label:")
file.writelines(str(len(entity_set)))
