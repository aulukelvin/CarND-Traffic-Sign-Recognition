## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/histogram.png "Histogram"
[image2]: ./images/classifiedimages.png "Classified images"
[image3]: ./images/batchsize.png "Batch size vs valid accuracy"
[image4]: ./images/mean.png "Mean normalization"
[image5]: ./testing_data/5/asd.jpeg "Traffic Sign 5"
[image6]: ./testing_data//21/21doubleCurve.jpeg "Traffic Sign 21"
[image7]: ./testing_data/22/bumpyroadcropped.jpeg "Traffic Sign 22"
[image8]: ./testing_data/23/23slipperyroad.jpeg "Traffic Sign 23"
[image9]: ./testing_data/29/29bicyclecrossing.jpeg "Traffic Sign 29"
[image10]: ./images/augmentation.png "augmentation.png"
[image11]: ./images/test1.png "test1"
[image12]: ./images/test2.png "test2"
[image13]: ./images/test3.png "test3"
[image14]: ./images/test4.png "test4"
[image15]: ./images/test5.png "test5"
[image16]: ./images/model.png "model"
[image17]: ./images/grayscale.png "grayscale"


### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

Here is a link to my [project code](https://github.com/aulukelvin/CarND-Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier_V2.ipynb)

### Data Set Summary & Exploration
The code for this step is contained in the second and the third code cell of the IPython notebook. I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset
The code for this step is contained in the fourth to sixth code cell of the IPython notebook. I used the matplotlib for plotting. 

Here is an exploratory visualization of the data set. I firstly checked the number of images of each class in the histogram as below. It shows that the distribution of the images are widely imbalanced. The most popular classes have nearly 2000 images per each in the training set, while the least popular classes in the training set only have around 200. I noticed the popularity of the classes has strong link to the accuracy of the classification in the later experiment. The second finding is the number of images in training set, validation set and test set looks in proportion over all classes. This is good because we don't need to worry about impact of biased data set. 

![Histogram of classes][image1]

Then I checked what the images in the data set looks like. I firstly explored the images in order and found out the images have closer number seem also have very high correlationship. I then randomly picked up 12 images for each class which show like the following. I noticed that the content and the quality still has wide variation in each class.

![Classified images][image2]

### Design and Test a Model Architecture

#### 1. Preprocessing

The code for this step is contained in the seventh code cell of the IPython notebook.

As a first step, I calculated the mean of each color channel, subtract the mean from color value and then divided by 180 so that I can convert the color value into float between -1 and 1. This transformation proved to be able to enhanced the performance by 2 percent. 

Then I top up with a histogram normalization which proved to be able to further improve the performance 1-2 percent.

I also tried grayscaling the input images and augmentation to boost the traing data set for each class to over 2600. The grayscale images and the histogram after augmentation are as the followings:

![Gray scale][image17]

![Augmented][image10]

#### 2. Normalization
I added L2 loss normalization at the very end of the project and found out L2 normalization can greatly reduce overfiting. The code piece of L2 normalization is as the following:

```pythong
vars   = tf.trainable_variables() 
lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if '_b' not in v.name ]) * 0.0003
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y) + lossL2
```

#### 3. Final model

The code for my final model is located in the tenth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x10x128 				    |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x256 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x256 				    |
| Flatten               |                                               |
| Fully connected		| input 6400, output 1024						|
| Fully connected		| input 1024, output 256						|
| Fully connected		| input 1024, output 128						|
| Softmax				| input 128, output 43							|

![Model][image16]

#### 4. Training

The code for training the model is located in the Eleventh cell of the ipython notebook. 

During model training, I used AdamOptimizer and use use training accuracy / loss, validation accuracy / loss to evaluate model performance.
When I train the model I noticed the result varies significantly from run to run. So I took 3-5 runs and use the average as the final result. This strategy was very useful when the model was not stable and the training was relatively faster. When the model getting more and more complicated, the time is also getting longer and longer so I finally turn back to use single run to evaluate model performance.

#### 5. Tuning process

Firstly I took the standard 5 layers LeNet model built in the last project as the base model, without any data processing, batch size 128. I got validation accuracy around 88%. 

Then I did a grid search to find the optimum batch size and found out batch size around 32 produces the best result, which enhanced the accuracy from 88% to above 90% as shown below:

![Batch size optimization][image3]

I then did mean normalization to boost the performance 3 percent as below, and then add histogram equalization which is also very efficient. I noticed that classes with fewer images produces worse result. So I tried to augment the data use keras datagenerator and tried to get better result by adjusting augment parameters but failed to find any significance. I finally removed the augmentation from the preprocessing.

What made me confused when playing with the first model is I saw the training accuracy is very close to 100%, training loss also less than 10 percent, but the valid accuracy was 10 percent below the trainging accuracy. Adding dropout only makes both the training and validation accruacy down, the big gap between the two accuracies still exists. 

I then slowly adding more capacity to the model by adding more convolutional layer and fully connected layers, increasing number of kernels, change filter size etc, and slightly add dropout to depress overfitting. I find out more complicated model can produce better result but it's getting harder and harder to adjust the model and the training time is getting longer and longer. I'd like to try pre-trained models like VGG16 but I didn't find a suitable one so I decide to leave it to the following Keras project because I know Keras has very handy built-in pre-trained models. 

![Preprocessing][image4]

Finally I added the L2 loss to the model and surprisingly found out the L2 is doing good job reducing overfitting. I noticed the gap between training and validation loss is much closer and training accuracy also increased a little bit. 

What I have learnt from this empirical experience is how to evaluate the performance of the model, how to decide when to add more capacity, when to add dropout, when to change learning rate, etc. I found besides the accuracy figures, the loss data can also be very useful for evaluating the progress. 

My final model results were:
* Training set accuracy = 0.997, training loss = 0.216;
* Validation set accuracy =  0.968, training loss = 0.336
* Test Accuracy = 0.951, Loss = 0.430

## Test a Model on New Images

#### 1. Random image test
To test the performance of the model, especially its general performance on least trained classes and difficult scenarios, I deliberately chose five German traffic sign pictures from the web: one from the popular class 5 - '80 km/h' which has 1800 input data in traing set; all four others from the least popular classes - class 21, 22, 23 and 29 which have input training data range from 200 to 400. I then manually cropped them and use opencv resize() function to convert them into 32X32X3 images.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

Each class of testing images was put in a separated folder with the class number as the folder name and all testing images were put into respective class folders. The images have to be jpeg format with name extension of '.jpeg'. The loading program will load all the jpeg files and use folder name as the label. The images were then been histogram normalized and mean normalized just as other dataset. The code for loading, preprocessing and exploring the testing images is in the code cell number 13.

The fourteenth cell of the jupyter notebook is a modified evaluate function producing cross entropy along with accuracy and loss. The fifteenth cell of the jupyter notebook is the part for running the model against the testing images.

The test result is as the following:

![Test image 1][image11]
![Test image 2][image12]
![Test image 3][image13]
![Test image 4][image14]
![Test image 5][image15]

The first image is a 80KM/h speed limit sign but the model mistakes it as 50KM/h speed limit sign and it's quite understandable. It might be difficult to classify because 50 and 80 are quite similar. I believe the model may needs more data and more capacity to clearly differentiate those classes. 

But the model was only able to correctly guess 1 of the 5 traffic signs, which is totally different from the accuracy of 95% on the given test data set. And for the only successful test, the top guess probability is only 34%, not so confident at all. And for the test image labeled 23, the model classified it as class 11. From the picture we can see that the two sign have similar pattern, so maybe this failure is also due to lack of training data. For the test cases labeled 21, and 29 the propability are flat which indicates the model has no idea what they are.

During exploration of the training images I noticed that most of the images were taken from several videos. The shape, position, lighting, and background of the images are very similar within groups. Inevitably, the images among training set, validation set, and test set are quite possibly highly related. So that instead of using the content of the sign panel as the input, the model may takes other noise such as the background to classify the images. And somehow the model may be really successful in using the leaking information rather than analyze the traffic sign itself. If we get new picture without such kinds of 'hint', the model will fail.

To further improve the model to make it general, maybe we can try to use opencv to crop out exactly the traffic sign itself to reduce the background as much as possible, and then we find a way to transform for example tilted sign to a regular one. We can also separate images into different groups according their similarity and avoid split images from same similarity group into both training and validation data set. By this way, we can reduce the opportunity of the 'cheating' in the model training. I think vastly augment data, use transformed data may help reduce the overfitting as well. Last and equally important, to reduce the error like mistakes 80 as 50, the model may need more capacity clearly identify the difference between the numbers. 

From the five test images I found that the quality of the input data has huge impact on the performance of the model. Four of the images from least popular classes yield very poor performance. I believe data augmentation may be a heal to the problem but I need to be more skillful to use that technique. 

