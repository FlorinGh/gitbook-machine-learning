# Machine Learning Projects

## [Self Driving Car - Advanced Lane Finding Project](https://fgheorghe.gitbook.io/machine-learning/lane-finding)

* Challenge:  write a software pipeline to identify the lane boundaries in a video taken while driving on a motorway
* Actions:
  * compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
  * apply a distortion correction on the video frames and save a corrected video
  * use colour transforms and gradients to create a threshold binary image
  * apply a perspective transform to rectify binary image \("birds-eye view"\)
  * detect lane pixels and fit to find the lane boundary
  * determine the curvature of the lane and vehicle position with respect to centre of curvature
  * warp the detected lane boundaries back onto the original perspective
  * output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Results: [https://github.com/FlorinGh/SelfDrivingCar-ND-pr4-Advanced-Lane-Lines](https://github.com/FlorinGh/SelfDrivingCar-ND-pr4-Advanced-Lane-Lines)

## [Self Driving Car - Vehicle Detection Project](https://fgheorghe.gitbook.io/machine-learning/vehicle-detection)

* Challenge:  write a software pipeline to detect vehicles in a video taken while driving on a motorway
* Actions: 
  * extract features from images using HOG \(histogram of oriented gradients\)
  * separate the images in train/test and train an SVM \(support vector machine\) classifier
  * implement a sliding window search and classify each window as vehicle or non-vehicle
  * output a video with the detected vehicles positions drawn as bounding boxes
* Results: [https://github.com/FlorinGh/SelfDrivingCar-ND-pr5-Vehicle-Detection](https://github.com/FlorinGh/SelfDrivingCar-ND-pr5-Vehicle-Detection) 

