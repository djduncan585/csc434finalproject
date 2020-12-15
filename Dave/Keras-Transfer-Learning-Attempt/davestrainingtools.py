import numpy
import keras
from keras import layers
import random
import sys
import warnings
import string
import random
import copy

#Generates string of newline separated items in randomized order.

#Parameter:
#textsource: A string containing lines separated by newline characters (\n).

#Returns:
# a string consisting of the textsource input with each \n separated line returned in random order.

def randomizedlines(textsource):
    textcopy = copy.deepcopy(textsource)
    reorderedlist = []
    textcopylist = textcopy.split('\n')

    while len(textcopylist) > 0:
        selection = random.randint(0, len(textcopylist) - 1)
        reorderedlist.append(textcopylist[selection] + '\n')
        del textcopylist[selection]

    reorderednames = ''.join(reorderedlist)
    return(reorderednames)


#Generates a string of text from an input source consisting of lines separated by the newline character (\n). Lines are returned in randomized order. Output is repeated to a specified length in characters.

#Parameters:
#listsource: A string containing multiple lines separated by \n.
#maxlength: An integer containing the length in characters of desired output.

#Returns:
#A string the length of maxlength containing lines in randomized order from the input "listsource".

def gensource(listsource, maxlength):
    modelcount = 0 #Characters generated of the training text
    reorderednamescount = 0 #Character of iteration of randomized name list
    sourcelist = []
    randomnames = randomizedlines(listsource)

    while modelcount < maxlength:
        if reorderednamescount >= len(randomnames):
            randomnames = randomizedlines(listsource)
            reorderednamescount = 0
        sourcelist.append(randomnames[reorderednamescount])
        reorderednamescount += 1;
        modelcount += 1;

    source = ''.join(sourcelist)
    return source
