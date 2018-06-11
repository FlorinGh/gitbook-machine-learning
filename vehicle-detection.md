# Vehicle Detection

## **Challenge**

Write a software pipeline to detect vehicles in a traffic video taken while driving on a motorway:

![](.gitbook/assets/project_video%20%281%29.gif)

## **Actions**

* Histogram of Oriented Gradients \(HOG\)
  * we have been provided with a set of images: 8968 various traffic images, 8792 images with cars;images are color, 64x64 pixels
  * using HOG and other techniques, extract features of these images
  * separate the images in train/test and train a SVM classifier
* Sliding Window Search
  * implemented a sliding window search and classify each window as vehicle or non-vehicle
  * run the function first on test images and afterwards against project video
* Video Implementation
  * output a video with the detected vehicles positions drawn as bounding boxes
  * implement a robust method to avoid false positives \(could be a heat map showing the location of repeated detection\)

### Tools

This project used a combination of Python, numpy, matplotlib, openCV, scikit-learn and moviepy; this is by definition a computer vision project. These tools are installed in a anaconda environment and ran in a Jupyter notebook.

The complete project implementation is available here: [https://github.com/FlorinGh/SelfDrivingCar-ND-pr4-Advanced-Lane-Lines](https://github.com/FlorinGh/SelfDrivingCar-ND-pr4-Advanced-Lane-Lines).

















## **Results**

All steps described above were captured in a pipeline; applying it over the frames of the traffic video renders the following result:

![](.gitbook/assets/project_video_output%20%281%29.gif)

For more details on this project visit the following github repository: [https://github.com/FlorinGh/SelfDrivingCar-ND-pr5-Vehicle-Detection](https://github.com/FlorinGh/SelfDrivingCar-ND-pr5-Vehicle-Detection).

