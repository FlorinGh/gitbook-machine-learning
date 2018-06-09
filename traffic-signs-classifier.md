# Traffic Signs Classifier

## Challenge

Train a convolution neural network to classify traffic signs images using the [German Traffic Sign Data set](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset); with the trained model classify traffic signs from the web.

![German Traffic Sign Data Set Classes](.gitbook/assets/traffic_signs%20%281%29.jpg)

## Actions

* install anaconda, setup an environment, install pandas, pickle,  tensorflow and openCV
* explore, summarise and visualise the data set
* design, train and test a model architecture
* use the model to make predictions on new images
* analyse the softmax probabilities of the new images

### Data Set Summary & Exploration

I used the python to find information about the traffic signs data set:

```python
# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of testing examples
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# How many unique classes/labels are in the dataset?
n_classes = 1 + y_train.max()

print("Number of training examples = ", n_train)
print("Number of validation examples = ", n_validation)
print("Number of testing examples = ", n_test)
print("Image data shape = ", image_shape)
print("Number of classes = ", n_classes)
```

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 pixels
* The number of unique classes/labels in the data set is 43

**2. Include an exploratory visualization of the dataset.**

Here is an exploratory visualization of the data set. It is a bins chart showing how the data is distributed between the train, validation and test sets. This also shows how many example there are for each class; the distribution between classes is not uniform, therefore some of the classes will train better; on the other hand we can see the same distribution for the validation and test cases, which is good as these will not be biassed.

![](.gitbook/assets/dataset_visual%20%281%29.jpg)



#### Design and Test a Model Architecture

**1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?**

The process is based mainly on LeNet lab; given the training accuracy was below the accepted threshold, some improvements have been made. For trainig I used tensorflow-gpu 1.6.

As a first step, I decided to convert the images to grayscale because this improved the speed of the training. Grayscale by itself doesn't improve accuracy, but it helps in speeding up the proccess.

Here is an example of a traffic sign image before and after grayscaling.



As a second step, I normalized the image data because this proved to improve the accuracy with about 3.5%; this process also helps in speeding up training.

I didn't use augmentation to add more data; the data set felt large enough to achive good results.

As a last step in preprocess, the training dataset was suffled for a better distribution of classes.

**2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.**

My final model consisted of the following layers:

| Layer | Description |
| :---: | :---: |
| Input | 32x32x1 Grayscale image |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
| RELU |  |
| Max pooling | 2x2 stride, valid padding, outputs 14x14x6 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU |  |
| Max pooling | 2x2 stride, valid padding, outputs 5x5x16 |
| Flatten | Outputs 400 |
| Dropout |  |
| Fully connected | Outputs 120 |
| RELU |  |
| Dropout |  |
| Fully connected | Outputs 84 |
| RELU |  |
| Dropout |  |
| Fully connected | Outputs 43 |
| Softmax | Outputs 43 |

**3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.**

To train the model, I used 50 epochs, bactch size of 128 and a learning rate of 0.001; dropout during training was set to 0.75; changed to 1.0 when evaluating against validation and test data sets. The optimizer used was AdamOptimizer.

**4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated.**

My final model results were:

* training set accuracy of 99.9%
* validation set accuracy of 95.5%
* test set accuracy of 95.0%

An iterative approach was chosen:

* Started from the LeNet lab architecture and tested it on the colored images, with an accuracy of 88.1% \(take these values with a pinch of salt as they change slightly when the dataset is re-shuffled\); I chose the LeNet because it was the single complete net that I've used successfuly before.
* This architecture was actually created for simpler problem, MNIST, with only one chanel of data and only 10 classes; so it was expected to have a low accuracy.
* Trying to improve accuracy I first did a couple of studies to see the effect of changing the batch size and learning rate on the accuracy; decided that 128 and 0.001 were the best values.
* Then I switched to tensorflow-gpu and increased the number of epochs from 10 to 300 and this an improvement of about 2%, getting to 90.2%; while this helps, it is not enough
* Then I applied normalization, which increased the accuracy by another 3.6%, getting to 93.8% on the test data, and 95% on the validation data; these values are above the threshold for submitting, but I knew I can do a bit better.
* Applying grayscale didn't improve much the accuracy, I only won a bit of speed
* The last bit I added in is dropout; added this on all layer expect the fisrt and last, dropout probability was 0.75 for training and 1.0 for evaluations; this increased accuracy to 95.0% on test set.
* Lessons learned: colors don't matter much in the machine world; always normalize to make all numbers in the same range; make it harder to train \(use dropout\) in order to make in easier at the test

#### Test a Model on New Images

**1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.**

I used 12 traffic signs that were not in the original data set:





The fourth image might be difficult to classify because half of the sign is coverred in snow.

**2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set \(OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric\).**

Here are the results of the prediction:

| Image | Prediction |
| :---: | :---: |
| Go straight or left | Go straight or left |
| Keep right | Keep right |
| No entry | No entry |
| No vehicles | No vehicles |
| Speed limit \(50km/h\) | Speed limit \(70km/h\) |
| Yield | Yield |
| Road work | Road work |
| Speed limit \(20km/h\) | Speed limit \(20km/h\) |
| Ahead only | Ahead only |
| Stop | Stop |
| Speed limit \(50km/h\) | Speed limit \(50km/h\) |
| Priority road | Priority road |

The model was able to correctly guess 11 of the 12 traffic signs, which gives an accuracy of 91.7%. This compares favorably to the accuracy on the test set of 95.0%

**3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.**

For the 1st image the top five soft max probabilities were:

| Probability | Prediction |
| :---: | :---: |
| 1.00 | Go straight or left |
| 0.00 | Roundabout mandatory |
| 0.00 | Ahead only |
| 0.00 | Keep left |
| 0.00 | Speed limit \(30km/h\) |

For the 2nd image the top five soft max probabilities were:

| Probability | Prediction |
| :---: | :---: |
| 1.00 | Keep right |
| 0.00 | Yield |
| 0.00 | Turn left ahead |
| 0.00 | Priority road |
| 0.00 | Go straight or right |

For the 3rd image the top five soft max probabilities were:

| Probability | Prediction |
| :---: | :---: |
| 0.93 | No entry |
| 0.06 | Stop |
| 0.00 | Keep right |
| 0.00 | Speed limit \(120km/h\) |
| 0.00 | Roundabout mandatory |

For the 4th image the top five soft max probabilities were:

| Probability | Prediction |
| :---: | :---: |
| 1.00 | No vehicles |
| 0.00 | No passing |
| 0.00 | Speed limit \(120km/h\) |
| 0.00 | Speed limit \(50km/h\) |
| 0.00 | Speed limit \(70km/h\) |

For the 5th image the top five soft max probabilities were:

| Probability | Prediction |
| :---: | :---: |
| 0.78 | Speed limit \(70km/h\) |
| 0.17 | Speed limit \(30km/h\) |
| 0.03 | Speed limit \(50km/h\) |
| 0.02 | Keep right |
| 0.00 | Go straight or left |

For the 6th image the top five soft max probabilities were:

| Probability | Prediction |
| :---: | :---: |
| 1.00 | Yield |
| 0.00 | Priority road |
| 0.00 | No passing |
| 0.00 | Ahead only |
| 0.00 | Road work |

For the 7th image the top five soft max probabilities were:

| Probability | Prediction |
| :---: | :---: |
| 1.00 | Road work |
| 0.00 | Dangerous curve to the right |
| 0.00 | Bumpy road |
| 0.00 | Beware of ice/snow |
| 0.00 | Bicycles crossing |

For the 8th image the top five soft max probabilities were:

| Probability | Prediction |
| :---: | :---: |
| 0.90 | Speed limit \(20km/h\) |
| 0.07 | Speed limit \(30km/h\) |
| 0.02 | Roundabout mandatory |
| 0.01 | Speed limit \(120km/h\) |
| 0.01 | Speed limit \(70km/h\) |

For the 9th image the top five soft max probabilities were:

| Probability | Prediction |
| :---: | :---: |
| 1.00 | Ahead only |
| 0.00 | Go straight or right |
| 0.00 | Turn left ahead |
| 0.00 | No vehicles |
| 0.00 | Speed limit \(60km/h\) |

For the 10th image the top five soft max probabilities were:

| Probability | Prediction |
| :---: | :---: |
| 0.60 | Stop |
| 0.07 | Speed limit \(30km/h\) |
| 0.06 | Keep right |
| 0.05 | Turn right ahead |
| 0.03 | Keep left |

For the 11th image the top five soft max probabilities were:

| Probability | Prediction |
| :---: | :---: |
| 1.00 | Speed limit \(50km/h\) |
| 0.00 | Speed limit \(30km/h\) |
| 0.00 | Speed limit \(80km/h\) |
| 0.00 | Roundabout mandatory |
| 0.00 | Speed limit \(60km/h\) |

For the 12th image the top five soft max probabilities were: \[12, 14, 2, 13, 38\]

| Probability | Prediction |
| :---: | :---: |
| 1.00 | Priority road |
| 0.00 | Stop |
| 0.00 | Speed limit \(50km/h\) |
| 0.00 | Yield |
| 0.00 | Keep right |

#### 

#### Optional: Visualizing the Neural Network

**1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?**

While neural networks can be a great learning device they are often referred to as a black box. This is not entirely true as we can look under the hood to see how the data evolves from one layer to another; tensor has the power to map all action under names and we can plot each action to have an idea of what is important at each layer.

Layer to focusses on the complete image taking into account a lot of details; as we goo deeper in the network, the amount of data is smaller for each image and fewer and fewer details are considered.

### 

![](https://github.com/FlorinGh/gitbook-machine-learning/tree/2cc3c9aedecf0c5bc08b2dfd0af6eae9389a2b3e/.gitbook/assets/tsc_traffic_signs.jpg)

### 

## Results

[https://github.com/FlorinGh/SelfDrivingCar-ND-pr2-Traffic-Signs-Classifier](https://github.com/FlorinGh/SelfDrivingCar-ND-pr2-Traffic-Signs-Classifier)

