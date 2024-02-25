# Yoga Pose Detection

Welcome to the Yoga Pose Detection project!

## About

This project is built using machine learning. We have used our custom model to train different yoga poses.

## Approach used to build custom model

1. To train our model, we have used dense layer which has activation function tanh.
2. To calculate loss value, we have used categorical cross-entropy loss.
3. The input to the model is landmarks detected from the pose.
4. For optimization algorithm, RMSProp(Root Mean Square Propogation) is used.


## Features

- Real-time pose detection
- Classification of yoga poses
- User-friendly interface
- Shows information about different yoga poses


## Pre-requists to run project
- Run env_sit/Scripts/activate.bat command for django inside projet root folder.

## Steps to run the project
- First run data_collection.py
        - In this file we can able to generate dataset dynamically
- Second run data_training.py
        - In this file,training of the model is done.
- Then you will able to run the django project.

## Screenshot
1. Accuracy and Loss - For Now, We Have A Small Dataset so based on that graph is displayed
   ![Accuracy Loss](https://github.com/Prachee03/Yoga_pose_detection/assets/130729921/1493ca55-2f58-4200-9dc7-9d8197ee5c01)
   
3. Confusion Matrix
   ![Confusion_Matrix](https://github.com/Prachee03/Yoga_pose_detection/assets/130729921/42db5b39-35da-40db-a1bf-67d42088a47a)
   
## Installation


To install the project, follow these steps:

1. Clone the repository: `git clone https://github.com/Prachee03/Yoga_pose_detection.git`
2. Install dependencies: `npm install`
3. Run the application: `npm start`

## Usage

Once the project is set up, you can start using it to detect yoga poses.

## Contributing

Contributions are welcomed! If you have any ideas for improvement or bug fixing, feel free to submit a pull request.
