import math
import cv2
import numpy as np 
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt 

# # Initialize mediapose class
mp_pose = mp.solutions.pose

# # setting up the funtion
pose = mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.3,model_complexity=2)

# # Initialize mediapipe drawing class
mp_drawing = mp.solutions.drawing_utils

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

def calculateAngle(landmark1, landmark2, landmark3):
    # get the required landmarks coordinate
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # calculate the angle between the three points
    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

    # if angle is negative add 360 to it.
    if angle < 0:
        angle += 360

    return angle    


def classifyPose(landmarks, output_image, display=False):
    label = 'Unknown Pose'
    color = (0, 0, 255)
    
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    if left_elbow_angle > 160 and left_elbow_angle < 195 and right_elbow_angle > 160 and right_elbow_angle < 195:
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                label = "Warrior II Pose"

    # T - Pose
    if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
        label = "T-Pose"

    # Tree - Pose
    if left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
        if left_knee_angle > 315 and left_knee_angle < 335 and right_knee_angle > 315 and right_knee_angle < 335:
            label = "Tree - Pose"

    if label != 'Unknown Pose':
        color = (0, 255, 0)

    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    if display:
        plt.figure(figsize=[20, 20])
        plt.imshow(output_image[:, :, ::-1])  # Display the output image
        plt.title("Output Image")
        plt.axis('off')
        plt.show()  # Show the plot
    else:
        return output_image, label
