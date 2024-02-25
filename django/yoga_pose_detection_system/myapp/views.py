from django.shortcuts import render
import os
import math
import cv2
import numpy as np 
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from django.http import JsonResponse
from keras.models import load_model

import tkinter as tk

root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()


def about(request):
    return render(request, 'about.html')

def poses(request):
    context = {
        'description': 'Different types of yoga poses'
    }
    return render(request, 'poses.html', context)

def index(request):
    return render(request, 'index.html')

def detect(request):
    vedio_capture()
    return render(request,'detect.html')

def vedio_capture(request):
    def inFrame(lst):
        if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
            return True 
        return False
   
    model  = load_model("E:/project/django/django/yoga_pose_detection_system/myapp/model.h5")
    label = np.load("E:/project/django/django/yoga_pose_detection_system/myapp/labels.npy")

    holistic = mp.solutions.pose
    holis = holistic.Pose()
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while cap.isOpened():  # Check if the capture object is open
        lst = []

        ret, frm = cap.read()

        if not ret:  # If frame is not retrieved, break from loop
            break

        window = np.zeros((screen_height, screen_width, 3), dtype="uint8")

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        frm = cv2.blur(frm, (4,4))
        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1,-1)

            p = model.predict(lst)
            pred = label[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                cv2.putText(window, pred , (180,180),cv2.FONT_ITALIC, 1.3, (0,255,0),2)
            else:
                cv2.putText(window, "Asana is either wrong not trained" , (100,180),cv2.FONT_ITALIC, 1.8, (0,0,255),3)

        else: 
            cv2.putText(frm, "Make Sure Full body visible", (100,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)

            
        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                connection_drawing_spec=drawing.DrawingSpec(color=(255,255,255), thickness=6 ),
                                 landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), circle_radius=3, thickness=3))

        resized_frame = cv2.resize(frm, (1280, 720))  # Resize the frame to your desired dimensions

        # Calculate the position to center the resized frame
        top_left_x = (screen_width - 1280) // 2
        top_left_y = (screen_height - 720) // 2

        window[top_left_y:top_left_y+720, top_left_x:top_left_x+1280, :] = resized_frame

        cv2.imshow("window", window)

        if cv2.waitKey(1) == 27:
            break  # Exit the loop if ESC key is pressed

    cv2.destroyAllWindows()
    cap.release()  # Release the webcam resources

    return render(request, 'detect.html')