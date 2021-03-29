## Face_mask_detector_1
Note: 
## We have not implemented live video streaming part. 
### First run face_1.py, it will preprocess the data and store the data in "data.npy" file. The target values will be stored in "target.npy" file.

### Second run face_2.py, it will load the "data.npy" and "target.npy" file and train the model. It will save the best model based on the accuracy of epochs. 

### Third run face_3.py, it will load the best saved model and use it to predict the two given images.
```
Here, we have not used data augmentation technique.
```
