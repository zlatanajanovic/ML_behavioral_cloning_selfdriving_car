#**Behavioral Cloning** 

##Writeup



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/middle_center.jpg "Model Visualization"
[image2]: ./examples/middle_right.jpg "Grayscaling"
[image3]: ./examples/middle_left.jpg "Recovery Image"
[image4]: ./examples/recovery1.jpg "Recovery Image"
[image5]: ./examples/recovery2.jpg "Recovery Image"
[image6]: ./examples/recovery3.jpg "Recovery Image"
[image7]: ./examples/flip1.jpg "Normal Image"
[image8]: ./examples/flip2.jpg "Flipped Image"
[image9]: ./examples/middle_center_cropped.jpg "Cropped image"
[image10]: ./examples/training1.png "Training with driving in the middle of the road"
[image11]: ./examples/training2.png "Additional training with recovery data"
[image12]: ./examples/training3.png "Additional training with driving in the middle of the road"

## Rubric Points
###Here I will consider the [rubric points]individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* model_fine_tuning_recovery.py containing the script to load and fine tune the model with central camera data for recovery
* model_fine_tuning.py containing the script to load and fine tune the model with driving data in the middle of the road

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 78-99). Starting point was NDVIDIA architecture with one Conv layer less.

The model includes RELU layers to introduce nonlinearity (code line 84, 86, 88, 90), and the data is normalized in the model using a Keras lambda layer (code line 80), and the data is cropped to the region of interest (code line 82. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 94, 96, 98). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 18). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 103).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and data augmentation by flipping data and using different cameras.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to develop preprocessing procedure on a simple model (model_simple.py). This model has 2 convolutional and one fully connected layer. This approach was choosen because of the time needed for training, and it showed to be useful in initial debugging.

This model had high mean squared error but it was useful to run code without errors. The next step was to improve model with variation of NVIDIA model. Variation was necessary because of different image shape

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. To combat the overfitting, I modified the model so that dropout layers are added. I found that my first model had a low mean squared error on the training and validation set.

![alt text][image10]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I was fine tuning model with some data from recovery manuevers (model_fine_tuning_recovery.py). The mean squared error after this training was high. The reason could be that data is not very consistent, manuevers are cut, and very much different from initial middle road driving. This improved driving in simulator. The vehicle was able to pass critical spots, but it was not smooth in the middle anymore. 

![alt text][image11]

To improve this, additional fine tuning with data on the middle of the road was executed(model_fine_tuning.py). 

![alt text][image12]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

Output of command:
```sh
print(model.summary())
```
is:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to

====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 318, 3)    0           lambda_1[0][0
]
________________________________________________________________________________
____________________
convolution2d_1 (Convolution2D)  (None, 61, 314, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 30, 157, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 26, 153, 36)   21636       maxpooling2d_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 13, 76, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 9, 72, 48)     43248       maxpooling2d_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 4, 36, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 2, 34, 64)     27712       maxpooling2d_3[0][0]
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 1, 17, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1088)          0           maxpooling2d_4[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           108900      flatten_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_3[0][0]
====================================================================================================
Total params: 208,891
Trainable params: 208,891
Non-trainable params: 0
____________________________________________________________________________________________________


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded one laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I used also data from other colleauges recorded with analog joystick, which is much smoother. The vehicle was recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the middle of the road. These images show what a recovery looks like starting from left :

![alt text][image4]

![alt text][image5]

![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would add more data and make even distribution of left and right turns as the track is mostly with left curves. For example, here is an image that has then been flipped:

![alt text][image7]
![alt text][image8]

To augment the data sat, I also used right and left camera images with steering angle ofset so that right camera steers left and left camera steers left.

![alt text][image1]

![alt text][image2]

![alt text][image3]

After that I cropped images in my model to get only region of interest. For example, here is an image that has then been cropped:

![alt text][image9]

After the collection process, I had 74640 data points for middle of the road driving and 2796 data for recovery manuevers.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by overfitting which started afterwards. I used an adam optimizer so that manually training the learning rate wasn't necessary.
