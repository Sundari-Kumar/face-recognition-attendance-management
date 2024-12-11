import cv2
import numpy as np
import os

def trainimg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create the face recognizer model
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Load the face detector
    
    try:
        faces, Id = getImagesAndLabels("TrainingImage")  # Get images and their labels
    except Exception as e:
        print('Please make "TrainingImage" folder and put Images')
        return

    recognizer.train(faces, np.array(Id))  # Train the recognizer with the collected faces and IDs

    try:
        # Save the trained model to the specified path
        recognizer.save("TrainingImageLabel/Trainner.yml")
    except Exception as e:
        print('Please make "TrainingImageLabel" folder to save the model')
        return

    print("Model Trained and saved successfully as 'Trainner.yml'")
