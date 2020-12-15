asciidict = {'\n': 0}
counter = len(asciidict.keys())
for i in range(32, 127):
    asciidict[chr(i)] = counter
    counter += 1
print(asciidict)