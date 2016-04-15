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

threshold = 0.001


def prepareData(num_of_affected_days, num_of_training_files):
	
	last_today_closed = 0

	rows = []
	index = []
	file_article_string = ""


	for day in range(2 + num_of_affected_days, num_of_training_files + 1):
    	day_number = int(day)

    	for cert_day in range(day_number, day_number - num_of_affected_days, -1):

        	dataset = json.load(
            	open(article_directory + str(cert_day) + ".json", 'r'))["dataset"]
        	# print dataset["articles"]

        	if(cert_day == day_number):
            	date = dataset["date"]

        	# take the last day's date to create labels accordingly

        	curr_string = '\n'.join(dataset["articles"])

        	# Transform last string by appending 1 to each word that string
        	# contains
        	if(cert_day == day_number - num_of_affected_days + 1):
            	curr_string = addone.addone(curr_string)
        	file_article_string += curr_string
        	# sys.stdout.write(str(cert_day) + ' ')
    	# print ""
    	# find the closing price of today
    	if os.path.isfile(price_directory + date + "_DJI.json"):
        	today_close = json.load(
            	open(price_directory + date + "_DJI.json", 'r'))["Close"]
    	# today_close = today_prices["Close"]
    	else:
        	today_close = last_today_closed
    	last_today_closed = today_close

    	# get one day after the date of articles by parsing the date and adding
    	# one day
    	tomorrow = datetime.datetime.strptime(
        	date, "%Y-%m-%d") + datetime.timedelta(days=1)
    	# find the closing price of tomorrow or one of the following days (in case
    	# tomorrow is a holiday)
    	count = 0
    	while not os.path.isfile(price_directory + str(tomorrow)[0:10] + "_DJI.json"):
        	# print tomorrow
        	# print price_directory + str(tomorrow)[0:10] + "_DJI.json "
        	tomorrow += datetime.timedelta(days=1)
        	count += 1
        	if count == 5:
            	exit()
    	moro_prices = json.load(
        	open(price_directory + str(tomorrow)[0:10] + "_DJI.json", 'r'))
    	moro_close = moro_prices["Close"]

    	change = (moro_close - today_close) / today_close

    	movement = ""

    	if change < -1 * threshold:
        	movement = "falling"
    	elif change < threshold:
        	movement = "neutral"
    	else:
        	movement = "rising"

    	# print "Today close: " + str(today_close) + " -> Moro close: " +
    	# str(moro_close) + " Movement: " + movement

    	rows.append({"text": file_article_string, "class": movement})
    	index.append(day)

	data_frame = DataFrame(rows, index=index)

	data_frame = data_frame.reindex(numpy.random.permutation(data_frame.index))

    return data_frame
