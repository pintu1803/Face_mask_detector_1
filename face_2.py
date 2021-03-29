

#Pintu, 181CO139
#29-03-2021
#Objective: Face mask detector using cnn.
#this is 2nd program.

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

#load the saved data and split it.
data=np.load('data.npy')
target=np.load('target.npy')
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1) #other parameters can be passed.

#create the model
model=Sequential()
model.add(Conv2D(200, (3,3), input_shape=data.shape[1:], activation='relu'))
model.add(MaxPooling2D((2,2)))
#first cnn layer.
model.add(Conv2D(100, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
#second cnn layer.
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

#compiling the model.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#checkpoint
checkpoint=ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

#train the model.
history=model.fit(x_train, y_train, epochs=15, callbacks=[checkpoint], validation_split=0.2)

#plot the training and validation loss.
plt.plot(history.history['loss'], 'r', label='Training loss')
plt.plot(history.history['val_loss'], 'b', label='Validation loss')
plt.xlabel('#Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plot the training and validation accuracy.
plt.plot(history.history['accuracy'], 'r', label='Training accuracy')
plt.plot(history.history['val_accuracy'], 'b', label='Validation accuracy')
plt.xlabel('#Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#evaluate the model.
print(model.evaluate(x_test, y_test))
