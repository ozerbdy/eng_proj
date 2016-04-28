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


def clssfyWith(real_values, example_counts, counts, targets, classifier):
    print "parameters: " + str(classifier.get_params())

    classifier.fit(counts, targets)

    predictions = classifier.predict(example_counts)

    correct_count = 0

    pred_length = len(predictions)

    for n in range(0, pred_length):
        if real_values[n] == predictions[n]: correct_count += 1

    print "Number of Correct Predictions:" + str(correct_count)
    print "Total Number of Predictions:" + str(pred_length)
    return


def prepareData(start_day, num_of_affected_days, num_of_training_files):
    
    
    datasetArr = {}
    for day in range(start_day + num_of_affected_days, num_of_training_files):
        dataset = json.load(
                open(article_directory + str(day) + ".json", 'r'))["dataset"]
        #print "new day", day
        datasetArr[day] = dataset

    print "Each \".\"(dot) represents 200 days"
    counter = 0

    last_today_closed = 0

    rows = []
    index = []

    

    for day in range(start_day + num_of_affected_days, num_of_training_files):
        if counter % 200 == 199 : 
            print '.'
        file_article_string = ""
        #print 'day', day
        day_number = int(day)
        for cert_day in range(day_number, day_number - num_of_affected_days, -1):
            #print 'certain day', cert_day
            dataset = datasetArr[day]
            #dataset = json.load(
            #    open(article_directory + str(cert_day) + ".json", 'r'))["dataset"]
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
        elif counter < 2 and last_today_closed == 0:
            continue
        else:
            today_close = last_today_closed
        last_today_closed = today_close

        #print today_close
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
        counter += 1

    print "\nTotal number of days prepared", counter

    data_frame = DataFrame(rows, index=index)

    data_frame = data_frame.reindex(numpy.random.permutation(data_frame.index))

    return data_frame


######################
# INITIAL CONDITIONS #
######################

number_of_files = len(os.listdir(article_directory))

print "Number of files", number_of_files

num_of_training_files = number_of_files * 9 / 10

print "Number of training files", num_of_training_files

num_of_affected_days = 3 #up to specified number of days 

for days in range(1, num_of_affected_days + 1):

    print "Number of affected days", days

    start_day = 2

    training_data_frame = prepareData(start_day, days, num_of_training_files)

    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(training_data_frame['text'].values)

    ##print "Features are extracted"



    targets = training_data_frame['class'].values


    test_data_frame = prepareData(num_of_training_files, days, number_of_files)

    example_counts = count_vectorizer.transform(test_data_frame["text"])

    real_values = test_data_frame["class"].values

    classifiersArray = []


    min_samples_splits = [5,7,10]
    max_depths = [5,10,20]
    naive_bayes_alphas = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 15.0, 25.0]
    n_estimators = [10, 20, 30, 40]
    Cs = [2.0, 4.0, 8.0, 20.0]

    #classifiers = []



    for i in naive_bayes_alphas:
        classifiersArray.append({'name': "Naive Bayes", 'classifier': MultinomialNB(alpha=i)})

    for i in n_estimators:
        classifiersArray.append({'name': "RandomForestClassifier", 'classifier': RandomForestClassifier(n_estimators=i)})

    for i in Cs:
        classifiersArray.append({'name': "Support Vector Machines", 'classifier': svm.SVC(C=i)})

    for i in min_samples_splits:   
        classifiersArray.append({'name' : 'DecisionTreeClassifier', 'classifier': DecisionTreeClassifier(min_samples_split=i)})

    for i in max_depths:
        classifiersArray.append({'name' : 'DecisionTreeClassifier', 'classifier': DecisionTreeClassifier(max_depth=i)})

    for clssfr in classifiersArray:
        

        print "\nClassifying with", clssfr['name']
        clssfyWith(real_values, example_counts, counts, targets, clssfr['classifier'])



"""
for clssfr, name in (
    (MultinomialNB(alpha=0.0), "Naive Bayes"),
    # default alpha value is 1.0
    (MultinomialNB(alpha=1.0), "Naive Bayes"),
    (MultinomialNB(alpha=2.0), "Naive Bayes"),
    (MultinomialNB(alpha=3.0), "Naive Bayes"),
    (MultinomialNB(alpha=5.0), "Naive Bayes"),
    (MultinomialNB(alpha=7.0), "Naive Bayes"),
    (MultinomialNB(alpha=15.0), "Naive Bayes"),
    (MultinomialNB(alpha=25.0), "Naive Bayes"),
    # default n_estimators value 10
    (RandomForestClassifier(n_estimators=10), "Random Forest Classifier"),
    (RandomForestClassifier(n_estimators=20), "Random Forest Classifier"),
    (RandomForestClassifier(n_estimators=30), "Random Forest Classifier"),
    (RandomForestClassifier(n_estimators=30, verbose=5), "Random Forest Classifier"),
    (RandomForestClassifier(n_estimators=30, verbose=20), "Random Forest Classifier"),
    (RandomForestClassifier(n_estimators=40, n_jobs=-1), "Random Forest Classifier"),
    (RandomForestClassifier(n_estimators=40), "Random Forest Classifier"),
    (DecisionTreeClassifier(), "Decision Tree Classifier"),
    (svm.SVC(), "Support Vector Machines"),
    (svm.SVC(kernel='linear'), "Support Vector Machines"),
    (svm.SVC(C=2.0), "Support Vector Machines"),
    (svm.SVC(C=4.0), "Support Vector Machines"),
    (svm.SVC(C=8.0), "Support Vector Machines"),
    (svm.SVC(C=20.0), "Support Vector Machines"),
    ):
    print "\n"+name
    clssfyWith(real_values, example_counts, counts, targets, clssfr)

"""



