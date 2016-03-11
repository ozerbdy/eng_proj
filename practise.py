import os
import json
import numpy
import os.path
from time import time
from pandas import DataFrame
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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


#for filename in os.listdir(article_directory): ??? or i may sort the list
for day in range(2, training_files):
	dataset = json.load(open(article_directory + str(day) + ".json" , 'r'))["dataset"]
	file_article_string = '\n'.join(dataset["articles"])


	#print file_article_string 
	date = dataset["date"]
	print date + "\n\n+++++\n\n+++++\n\n+++++\n\n" + str(day) + ". day"
	
	#find the closing price of today
	if os.path.isfile(price_directory + date + "_DJI.json"):
		today_close = json.load(open(price_directory + date + "_DJI.json", 'r'))["Close"]
	#today_close = today_prices["Close"]
	else:
		today_close = last_today_closed 
	last_today_closed = today_close
	
	#get one day after the date of articles by parsing the date and adding one day
	tomorrow = datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=1)
	#find the closing price of tomorrow or one of the following days (in case tomorrow is a holiday)
	count = 0
	while not os.path.isfile(price_directory + str(tomorrow)[0:10] + "_DJI.json"):
		print tomorrow
		print price_directory + str(tomorrow)[0:10] + "_DJI.json "
		tomorrow += datetime.timedelta(days=1)
		count += 1
		if count == 5: 
			exit()
	moro_prices = json.load(open(price_directory + str(tomorrow)[0:10] + "_DJI.json", 'r'))
	moro_close = moro_prices["Close"]

	change = (moro_close - today_close)/today_close

	movement = ""
	

	if change < -1*threshold:
		movement = "falling"
	elif change < threshold:
		movement = "neutral"
	else: 
		movement = "rising"

	print "Today close: " + str(today_close) + " -> Moro close: " + str(moro_close) + " Movement: " + movement

	rows.append({ "text" : file_article_string , "class" : movement})
	index.append(day)


data_frame = DataFrame(rows, index=index)

data_frame = data_frame.reindex(numpy.random.permutation(data_frame.index))

  #---------------------#
 # EXTRACTING FEATURES #
#---------------------#

count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(data_frame['text'].values)

print "Features are extracted"


  #----------------------#
 # CLASSIFYING ARTICLES #
#----------------------#

#classifierNB = MultinomialNB()
#classifierSVM = svm.SVC()

targets = data_frame['class'].values

#classifierNB.fit(counts, targets)
#classifierSVM.fit(counts, targets)


def clssfyWith(classifier, cluster):
	
	if cluster : 
		predictions = classifier.fit_predict(counts)

	else : 
		classifier.fit(counts, targets)

		predictions = classifier.predict(example_counts)

	correct_count = 0

	pred_length = len(predictions)

	for n in range(0, pred_length):
		if real_values[n] == predictions[n] : correct_count += 1

	print "Number of Correct Predictions:" + str(correct_count)
	print "Total Number of Predictions:" + str(pred_length)
	return 



print "Articles are classified"


"""
examples = ["Prices will go up good nice positive", "This is a bad day down worse"]

example_counts = count_vectorizer.transform(examples)

predictions = classifier.predict(example_counts)

print "Predictions:"
print predictions 

"""

    #--------------------------------------------------#
   #                                                  #
  #                TESTING CLASSIFIER                #
 #                                                  #
#--------------------------------------------------#



#  PREPARING TEST DATA  #


test_rows = []
test_index = []




for day in range(training_files, number_of_files):
	dataset = json.load(open(article_directory + str(day) + ".json" , 'r'))["dataset"]
	file_article_string = '\n'.join(dataset["articles"])


	#print file_article_string 
	date = dataset["date"]
	print date + "\n\n+++++\n\n+++++\n\n+++++\n\n" + str(day) + ". day"
	
	#find the closing price of today
	if os.path.isfile(price_directory + date + "_DJI.json"):
		today_close = json.load(open(price_directory + date + "_DJI.json", 'r'))["Close"]
	#today_close = today_prices["Close"]
	else:
		today_close = last_today_closed 
	last_today_closed = today_close
	
	#get one day after the date of articles by parsing the date and adding one day
	tomorrow = datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=1)
	#find the closing price of tomorrow or one of the following days (in case tomorrow is a holiday)
	count = 0
	while not os.path.isfile(price_directory + str(tomorrow)[0:10] + "_DJI.json"):
		print tomorrow
		print price_directory + str(tomorrow)[0:10] + "_DJI.json "
		tomorrow += datetime.timedelta(days=1)
		count += 1
		if count == 5: 
			exit()
	moro_prices = json.load(open(price_directory + str(tomorrow)[0:10] + "_DJI.json", 'r'))
	moro_close = moro_prices["Close"]

	change = (moro_close - today_close)/today_close

	movement = ""
	

	if change < -1*threshold:
		movement = "falling"
	elif change < threshold:
		movement = "neutral"
	else: 
		movement = "rising"

	print "Today close: " + str(today_close) + " -> Moro close: " + str(moro_close) + " Movement: " + movement

	test_rows.append({ "text" : file_article_string , "class" : movement})
	test_index.append(day)


test_data_frame = DataFrame(test_rows, index=test_index)


# EVALUATING TEST DATA


example_counts = count_vectorizer.transform(test_data_frame["text"])

real_values = test_data_frame["class"].values

"""
predictionsNB = classifierNB.predict(example_counts)

predictionsSVM = classifierSVM.predict(example_counts)


correct_count = 0

pred_lengthNB = len(predictionsNB)

for n in range(0,pred_lengthNB):
	#print "Real value: " + real_values[n] + " - PredictionNB: " + predictionsNB[n] 
	if real_values[n] == predictionsNB[n] : correct_count += 1

print "Number of Correct Predictions Naive Bayes: " + str(correct_count)
print "Total Number of Predictions Naive Bayes: " + str(pred_lengthNB)

correct_count = 0

pred_lengthSVM = len(predictionsSVM)

for n in range(0,pred_lengthSVM):
	#print "Real value: " + real_values[n] + " - PredictionSVM: " + predictionsNB[n] 
	if real_values[n] == predictionsSVM[n] : correct_count += 1

print "Number of Correct Predictions SVM: " + str(correct_count)
print "Total Number of Predictions SVM: " + str(pred_lengthSVM)

"""


#classifierNB = MultinomialNB()
#classifierSVM = svm.SVC()

for clssfr, cluster, name in (
	(MultinomialNB(), False, "Naive Bayes"),
	(RandomForestClassifier(), False, "Random Forest Classifier"),
	(DecisionTreeClassifier(), False, "Decision Tree Classifier"),
	#(KMeans(n_clusters=3), True, "KMeans with 3 Clusters"),
	#(KMeans(n_clusters=8), True, "KMeans with 8 Clusters"),
	(svm.SVC(), False, "Support Vector Machines"),):
	print "\n"+name
	clssfyWith(clssfr, cluster)

"""
clssfyWith(classifierNB)
clssfyWith(classifierSVM)
"""


#results -> ["rising", "falling"]

#file_article_string = ' '.join(file_article_array)

#for article in file_article_array:
#	file_article_string.append(article)


#http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html


#if not filename.endswith("_DJI.json"):






