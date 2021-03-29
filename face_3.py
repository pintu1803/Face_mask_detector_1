

#Pintu, 181CO139
#29-03-2021
#Objective: Face mask detector using cnn.
#this is 3rd program.

from keras.models import load_model
import cv2
import numpy as np

#load the model.
model=load_model('model-015.model')

#WE ARE GOING TO TEST ON TWO IMAGES.
#(1) read the image.
img=cv2.imread('test_photo.jpg')
#convert rgb image into grayscale.
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#make it 100x100 pixel image.
resized=cv2.resize(img, (100,100))
#normalize the image.
normalized=resized/255.0
#reshape the image, to pass it to the model for testing.
reshaped=np.reshape(normalized, (1,100,100,1))
#prediction made by model.
result=model.predict(reshaped)
#find label with higher probability.
label=np.argmax(result, axis=1)[0]
#create the label_dictionary.
label_dict={0: 'With_Mask', 1:'Without_Mask'}

#display the image.
cv2.imshow("Photo",resized)
cv2.waitKey()
#print the result.
print("Result=",label_dict[label])


#(2) read the image.
img=cv2.imread('pintu_with_mask.jpeg')
#convert rgb image into grayscale.
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#make it 100x100 pixel image.
resized=cv2.resize(img, (100,100))
#normalize the image.
normalized=resized/255.0
#reshape the image, to pass it to the model for testing.
reshaped=np.reshape(normalized, (1,100,100,1))
#prediction made by model.
result=model.predict(reshaped)
#find label with higher probability.
label=np.argmax(result, axis=1)[0]
#create the label_dictionary.
label_dict={0: 'With_Mask', 1:'Without_Mask'}

#display the image.
cv2.imshow("Photo",resized)
cv2.waitKey()
#print the result.
print("Result=",label_dict[label])