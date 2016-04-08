def addone(string):
	resultarr = []
	newstr = string.split()
	for thisstr in newstr:
		resultarr.append(thisstr + '1')
	return ' '.join(resultarr)



"""
thestring = "hello there\nwhat am i. ? doing -- . here=?"

print thestring
print addone(thestring)
"""