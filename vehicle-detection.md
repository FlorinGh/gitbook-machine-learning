# Vehicle Detection

## **Challenge**

Write a software pipeline to detect vehicles in a video taken while driving on a motorway:

![](.gitbook/assets/project_video%20%281%29.gif)

## **Actions**

* extract features from images using HOG \(histogram of oriented gradients\)
* separate the images in train/test and train an SVM \(support vector machine\) classifier
* implement a sliding window search and classify each window as vehicle or non-vehicle
* output a video with the detected vehicles positions drawn as bounding boxes

### Tools

This project used a combination of Python, numpy, matplotlib, openCV, scikit-learn and moviepy; this is by definition a computer vision project. These tools are installed in a anaconda environment and ran in a Jupyter notebook.

The complete project implementation is available here: [https://github.com/FlorinGh/SelfDrivingCar-ND-pr4-Advanced-Lane-Lines](https://github.com/FlorinGh/SelfDrivingCar-ND-pr4-Advanced-Lane-Lines).

















## **Results**

All steps described above were captured in a pipeline; applying it over the frames of the traffic video renders the following result:

![](.gitbook/assets/project_video_output%20%281%29.gif)

For more details on this project visit the following github repository: [https://github.com/FlorinGh/SelfDrivingCar-ND-pr5-Vehicle-Detection](https://github.com/FlorinGh/SelfDrivingCar-ND-pr5-Vehicle-Detection).

