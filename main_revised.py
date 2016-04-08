import os
import sys
import json
import addone
import numpy
import os.path
from time import time
from pandas import DataFrame
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import svm


article_directory = "dataset/training/"

price_directory = "dataset/price/"



rows = []
index = []

current_file = 2

threshold = 0.001

last_today_closed = 0

#file_article_string = ""

number_of_files = len(os.listdir(article_directory))

training_files = number_of_files * 9 / 10

num_of_affected_days = 3;

file_article_string = ""

for day in range(2 + num_of_affected_days, training_files + 1):
	day_number = int(day)
	file_article_string = ""
	for cert_day in range(day_number, day_number - num_of_affected_days, -1):
		dataset = json.load(open(article_directory + str(cert_day) + ".json" , 'r'))["dataset"]
		#print dataset["articles"]

		curr_string = '\n'.join(dataset["articles"])

		#Transform last string by appending 1 to each word that string contains  
		if(cert_day == day_number - num_of_affected_days + 1):
			curr_string = addone.addone(curr_string)
		file_article_string += curr_string
		#sys.stdout.write(str(cert_day) + ' ')
	#print ""

"""
rows.append({'text' : file_article_string})
index.append(day_number)

data_frame = DataFrame(rows, index=index)

count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(data_frame['text'].values)

print counts.toarray()
"""

#print file_article_string.split()

#print file_article_string 

"""
def addone(string):
	resultarr = []
	newstr = str.split(string)
	for thisstr in newstr:
		resultarr.append(thisstr + '1')
	return ' '.join(resultarr)
"""