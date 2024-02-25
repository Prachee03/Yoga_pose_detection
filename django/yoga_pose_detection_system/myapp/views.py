from django.shortcuts import render
import math
import cv2
import numpy as np 
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt 


def about(request):
    return render(request, 'about.html')

def poses(request):
    context = {
        'description': 'Different types of yoga poses'
    }
    return render(request, 'poses.html', context)


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

def index(request):
    return render(request, 'index.html')


def detectPose(image, pose, display=True):
    # Check if the image is loaded successfully
    if image is None:
        print("Error: Image not loaded.")
        return None, None

    # create a copy of the input image
    output_image = image.copy()

    # convert the image from BGR into RGB format
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the pose detection
    results = pose.process(imageRGB)

    # Retrieve height and width
    height, width, _ = image.shape

    # Initialize a list to store detected landmarks
    landmarks = []

    # check if any landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width) if landmark.HasField('z') else None))

    # check if original input image and resultant image are specified
    if display:
        # diplay both image
        plt.figure(figsize=[22, 22])
        plt.subplot(121);plt.imshow(image[:, :, ::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:, :, ::-1]);plt.title("Output Image");plt.axis('off');

        # Plot Pose landmarks in 3D
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    else:
        # Return the output image
        return output_image, landmarks


def pose_detection(request):
       if request.method == 'POST' and request.FILES['image']:
            image = request.FILES['image']
            processed_image, landmarks = detect_pose(image)
            return JsonResponse({'image': processed_image, 'landmarks': landmarks})        
       return render(request, 'index.html')


