from textgenrnn import textgenrnn
import sys

with open('DEMF-Movement Festival performers-v2.txt','r') as f_open:
    source = f_open.read()
#First parameter is number of lines (DJ names) generated
if len(sys.argv) >= 2:
    outputlen = int(sys.argv[1])
else:
    outputlen = 10
#Second parameter is temperature
if len(sys.argv) >= 3:
    t = float(sys.argv[2])
else:
    t = 0.2


model = textgenrnn()
model.load("textgenrnn_weights.hdf5")
outputlist = model.generate(n=outputlen, temperature = t, return_as_list=True)
for i in range(0, len(outputlist)):
    print(outputlist[i])
