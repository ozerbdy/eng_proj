import os
import json

results_directory = 'all/'

filenames = os.listdir(results_directory)




def takeTheNum(whole_string, stringToDelete):

	return int(whole_string.replace(stringToDelete, ''))

def takeTheRest(whole_string, stringToDelete):

	return whole_string.replace(stringToDelete, '').split('\n')[0]

def accuracy(aDict):
	return aDict['Number of Correct Predictions:'] / float(aDict['Total Number of Predictions:'])

introduction = [
'Number of files',
'Number of training files',
'Number of affected days',
'Total number of days prepared',
'number of falling is',
'number of neutral is',
'number of rising is',
'Total number of days prepared',
'number of falling is',
'number of neutral is',
'number of rising is',
]

classification = [
'Classifying with ', 
'parameters:',
'Number of Correct Predictions:',
'Total Number of Predictions:'
]

dataset = []

count = 0

limit = 22

sublimit = 4

newobj = {}

temp_arr = []


f = open(results_directory + filenames[0], 'r')

for filename in filenames:
	print filename
	for line in f:
		for key in introduction:
			if line.find(key) != -1 or line == '\n':
				break
		else:	
			curr_str = classification[count % 4]
			if count % 4 == 0 or count % 4 == 1:
				newobj[curr_str] = takeTheRest(line, curr_str)
			else:
				newobj[curr_str] = takeTheNum(line, curr_str)	
			#print line,#.split('\n')
			if count % 4 == 3:
				temp_arr.append(newobj)
				newobj = {}
			if count % 88 == 87:
				dataset.append(temp_arr)
				temp_arr = []
			count += 1



	index = 1

	for day_interval in dataset:
		print 'day interval',index, '\n-----------------------------------\n'
		for test in day_interval:
			print test[classification[0]],',' ,accuracy(test)
		print '\n------------------------------------\n'
		index += 1
	#print accuracy(dataset[0][0])



# 22 - 3 - 4





"""
for line in f:
	if line.find('Number of files') != -1:
		NewDay += 1
		intro = True
		dataset.append(newobj)
		current_key = introduction[intro_count]
		print dataset, NewDay
		dataset[NewDay][current_key] = takeTheNum(line, current_key)
	elif intro:
		intro_count += 1
		print intro_count
		current_key = introduction[intro_count]
		dataset[NewDay][current_key] = takeTheNum(line, current_key)
		if intro_count == len(introduction) - 1:
			intro = False
	else:
		intro = False
	
"""



#f = open('filename', 'r')

#first line is name of the file so ignore it



"""




"""