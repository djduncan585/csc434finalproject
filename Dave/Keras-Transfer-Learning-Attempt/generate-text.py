'''
generate-text.py [outputlength=500] [temperature=0.2]
Generates text from model "the_model" with prompts selected at random from "extended-dj-names-random-order.txt" (randomized list of DJ names from a list of performers at the Detroit Electronic Music Festival and Movement Festival from the years 2000 through 2019.) Output length defaults to 500 characters, and temperature 0.2. (see Francois Chollet's method from _Deep Learning with Python_, https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.1-text-generation-with-lstm.ipynb)
'''

from getopt import getopt
import numpy
import keras
from keras import layers
import random
import sys
import tensorflow as tf
import os
import warnings

#gets rid of divide by zero warnings
warnings.filterwarnings("ignore")

def sample(preds, temperature=1.0):
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)

maxlen = 20
#First parameter is length in chars
if len(sys.argv) >= 2:
    outputlength = int(sys.argv[1])
else:
    outputlength = 500
#Second parameter is temperature
if len(sys.argv) >= 3:
    temperature = float(sys.argv[2])
else:
    temperature = 0.2

source = open("extended-dj-names-random-order.txt").read().lower()

char_indices = {'\n': 0, ' ': 1, '!': 2, '"': 3, '#': 4, '$': 5, '%': 6, '&': 7, "'": 8, '(': 9, ')': 10, '*': 11, '+': 12, ',': 13, '-': 14, '.': 15, '/': 16, '0': 17, '1': 18, '2': 19, '3': 20, '4': 21, '5': 22, '6': 23, '7': 24, '8': 25, '9': 26, ':': 27, ';': 28, '<': 29, '=': 30, '>': 31, '?': 32, '@': 33, 'A': 34, 'B': 35, 'C': 36, 'D': 37, 'E': 38, 'F': 39, 'G': 40, 'H': 41, 'I': 42, 'J': 43, 'K': 44, 'L': 45, 'M': 46, 'N': 47, 'O': 48, 'P': 49, 'Q': 50, 'R': 51, 'S': 52, 'T': 53, 'U': 54, 'V': 55, 'W': 56, 'X': 57, 'Y': 58, 'Z': 59, '[': 60, '\\': 61, ']': 62, '^': 63, '_': 64, '`': 65, 'a': 66, 'b': 67, 'c': 68, 'd': 69, 'e': 70, 'f': 71, 'g': 72, 'h': 73, 'i': 74, 'j': 75, 'k': 76, 'l': 77, 'm': 78, 'n': 79, 'o': 80, 'p': 81, 'q': 82, 'r': 83, 's': 84, 't': 85, 'u': 86, 'v': 87, 'w': 88, 'x': 89, 'y': 90, 'z': 91, '{': 92, '|': 93, '}': 94, '~': 95}

chars = list(char_indices.keys())

model = keras.models.load_model("the_model")

start_index = random.randint(0, len(source) - maxlen - 1)
generated_text = source[start_index: start_index + maxlen]

for i in range(outputlength):
    sampled = numpy.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(generated_text):
        sampled[0, t, char_indices[char]] = 1.

    preds = model.predict(sampled, verbose=0)[0]
    next_index = sample(preds, temperature)
    next_char = chars[next_index]

    generated_text += next_char
    generated_text = generated_text[1:]

    sys.stdout.write(next_char)
    sys.stdout.flush()