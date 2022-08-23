#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import joblib
import re
import sys
import os

sys.path.append(os.path.abspath('../tools/'))
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer

from_sara = open('from_sara.txt', 'r')
from_chris = open('from_chris.txt', 'r')

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker

for (name, from_person) in [('sara', from_sara), ('chris', from_chris)]:
    for path in from_person:
        path = os.path.join('..', path[:-1])
        # print(path)
        email = open(path, 'r')

        # ## use parseOutText to extract the text from the opened email

        stemmed_email = parseOutText(email)

        # ## use str.replace() to remove any instances of the words
        # ## ["sara", "shackleton", "chris", "germani"]

        for e in ['sara', 'shackleton', 'chris', 'germani', 'sshacklensf', 'cgermannsf']:
            stemmed_email = stemmed_email.replace(e, '')

        # ## append the text to word_data

        word_data.append(stemmed_email)

        # ## append a 0 to from_data if email is from Sara, and 1 if email is from Chris

        if name == 'sara':
            from_data.append(0)
        else:
            from_data.append(1)

        email.close()

#print('Emails Processed')
from_sara.close()
from_chris.close()

joblib.dump(word_data, open('your_word_data.pkl', 'wb'))
joblib.dump(from_data, open('your_email_authors.pkl', 'wb'))

### in Part 4, do TfIdf vectorization here
vectorizer = TfidfVectorizer(stop_words="english")
word_data = vectorizer.fit_transform(word_data)
#print(vectorizer.get_feature_names_out()[34597])
