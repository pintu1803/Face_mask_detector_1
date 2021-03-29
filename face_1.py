

#Pintu, 181CO139
#29-03-2021
#Objective: Face mask detector using cnn.
#this is 1st program.

import cv2, os
import warnings
warnings.simplefilter("ignore")

data_path='dataset'
#extract the names of folder inside dataset folder.
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories, labels))
#print them.
print("Categories dictionary=",categories)
print("Labels in target attribute=",labels)
print("Label dictionary=",label_dict)

#we want to create two lists, each for image and label.
img_size=100
data=[]
target=[]
for category in categories:
    folder_path=os.path.join(data_path, category)
    img_names=os.listdir(folder_path)

    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        try:
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized=cv2.resize(gray, (img_size,img_size))
            data.append(resized)
            target.append(label_dict[category])
        except Exception as e:
            print("Exception:",e)

#convert the images in array.
import numpy as np
#normalize the pixel values in range[0-1]
#data type: list->ndarray
data=np.array(data)/255.0
data=np.reshape(data, (data.shape[0], img_size, img_size, 1))
#target type: list->ndarray
target=np.array(target)

#convert target values in categorical format.
from keras.utils import np_utils
new_target=np_utils.to_categorical(target)
# print(target[0], new_target[0],type(new_target))
np.save('data', data)
np.save('target', new_target)

