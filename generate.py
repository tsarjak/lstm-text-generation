import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import RMSprop
import sys

'''
model = Sequential()
model.add(LSTM(256, input_shape=(100, 1)))
model.add(Dropout(0.2))
model.add(Dense(52, activation='softmax'))
'''

model = Sequential()
model.add(LSTM(128, input_shape=(35, 1), return_sequences=True))
model.add(Dropout(0))
model.add(LSTM(128))
model.add(Dropout(0))
model.add(Dense(31, activation='softmax'))

trainerOpt = RMSprop(lr=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=trainerOpt)
filename = "new-modelOpt-updated-weights-10-1.2587.hdf5"
model.load_weights(filename)


completeText = open('Processed_2.txt').read().lower()
chars = sorted(list(set(completeText)))
charToInt = dict((c,i) for i,c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

pattern = "energetic. you are severe on us. it"
sort_sen = list(pattern)
pattern = [charToInt[value.lower()] for value in sort_sen]

print(pattern)
# generate characters
for i in range(400):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(31)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]