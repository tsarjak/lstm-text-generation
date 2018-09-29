import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import RMSprop

def loadText(filename):
	completeText = open(filename).read().lower()
	chars = sorted(list(set(completeText)))
	charToInt = dict((c,i) for i,c in enumerate(chars))


	totalChars = len(completeText)
	totalVocab = len(chars)


	seqLength = 35
	dataX = []
	dataY = []
	for i in range(0, totalChars - seqLength, 1):
		inputSeq = completeText[i:i + seqLength]
		outSeq = completeText[i + seqLength]
		dataX.append([charToInt[char] for char in inputSeq])
		dataY.append(charToInt[outSeq])

	nPatterns = len(dataX)
	print("Total Patterns: ", nPatterns) 

	return dataX, dataY, seqLength, totalVocab, nPatterns


def trainNet():
	dataX, dataY, seqLength, totalVocab, nPatterns = loadText('Processed_2.txt')
	X = np.reshape(dataX, (nPatterns, seqLength, 1))
	X = X / float(totalVocab)
	y = np_utils.to_categorical(dataY)

	print(y.shape[1])

	print(X.shape[1], X.shape[2])

	model = Sequential()
	model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
	#model.add(Dropout(0))
	model.add(LSTM(128))
	#model.add(Dropout(0))
	model.add(Dense(y.shape[1], activation='softmax'))

	trainerOpt = RMSprop(lr=0.00001)
	model.compile(loss='categorical_crossentropy', optimizer=trainerOpt)


	filenameLoad = "new-model-updated-weights-14-1.3703.hdf5"
	model.load_weights(filenameLoad)


	filepath="new-modelOpt-updated-weights-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]


	# fit the model
	model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

	

trainNet()


''' 
instead of length 100, length 50. - not changing 
Add another layer of lstm. done
use another dataset with atleast 4,00,000 patterns - done - now 4,97,000 patterns
train for 30 epochs at the least	will try
character vocabulary should be less than 35 not very sure
'''


'''
Update two - did five epochs without any decay and a learning rate of 0.001.
Try to decrease the learning rate and add decay to it.
was wrong the last time probably - need to increase the sequence length to make it learn more from the context
'''