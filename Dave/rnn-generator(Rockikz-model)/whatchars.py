import numpy as np

FILE_PATH = "training-performer-names.txt"
# read the data
text = open(FILE_PATH, encoding="utf-8").read()
textarray = []
for i in range(0, len(text)):
    textarray.append(text[i])
uniquechars = np.unique(textarray)
print(f"{len(textarray)} characters total.")
print(f"{len(uniquechars)} unique characters.")
for i in range(0, len(uniquechars)):
    print(uniquechars[i], ord(uniquechars[i]))
