import numpy
import keras
from keras import layers
import random
import sys
import string
import davestrainingtools as dtt

source1length = len(open("future-shock-filtered.txt").read().lower())
source2 = open("../../DEMF-Movement Festival performers-v2.txt").read().lower()
longerfile = []
longertext = dtt.gensource(source2, source1length)
filewrite = open("extended-dj-names-random-order.txt", "w")
filewrite.write(longertext)
filewrite.close()
