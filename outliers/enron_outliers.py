#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

s = []
b = []
for point in data:
    salary = point[0]
    s.append([salary])
    bonus = point[1]
    b.append([bonus])
    matplotlib.pyplot.scatter( salary, bonus )
    
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(s, b)

matplotlib.pyplot.plot(s, reg.predict(s))

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

