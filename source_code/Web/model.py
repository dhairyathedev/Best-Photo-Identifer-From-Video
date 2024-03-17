import gradio as gr
import os
from skimage import filters
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model

import tensorflow as tf

print(tf.__version__)

# Load models
model_emotion = load_model(r"face_emotion.h5")
attractive_model = load_model('model2.h5')

def calculate_exposure(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Calculate exposure using histogram
    exposure = np.sum(hist * np.arange(256)) / np.sum(hist)

    return exposure

def calculate_clarity(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate clarity using Laplacian filter
    clarity = filters.laplace(gray_image).var()

    return clarity

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (124, 124))
    feature = np.array(resized_image)
    feature = feature.reshape(1, 124, 124, 1)
    return feature / 255.0

def preprocess_image_for_attractive(image):
    image = cv2.resize(image, (218, 172))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    image = image / 255.0
    image = image.reshape(1, 172, 218, 3)
    return image

def get_blurrness_score(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm

def process_video(video_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Labels for emotions
    labels = {0: 'angry', 1: 'disgust', 2: 'fear',
              3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    # Counter for saved images

    image_count = 0
    i = 0

    # store all frames in a list
    frames = []
    happy_scores = []
    happy_frames = []
    happy_faces = []
    attractive_scores = []
    face_blur_scores = []

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        if i % 5 == 0:
            frames.append(frame)
        i += 1
    video_capture.release()

    # Loop through each frame of the video
    for frame in frames:
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            frame, scaleFactor=1.3, minNeighbors=5)
        # Iterate through each detected face
        if len(faces) > 1:
            happy_score = 0
            score=0
            attr_score = 0
            blur_score = 0
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                model_input = np.expand_dims(face_roi, axis=0)
                img_features = extract_features(face_roi)
                prediction = model_emotion.predict(img_features)
                outputlabel = labels[np.argmax(prediction)]
                print("Emotion: ", outputlabel)
                if outputlabel == 'happy':
                    happy_score+=prediction[0][3]
                    score+=1
                    img_attrac = preprocess_image_for_attractive(face_roi)
                    attractive_prediction = attractive_model.predict(img_attrac)
                    attr_score = attractive_prediction[0][0]
                    blur_score = attractive_prediction[0][1]
            if(score/len(faces)>0.5):
                happy_score/=len(faces)
                happy_frames.append(frame)
                print("Happy Score: ", happy_score)
                happy_scores.append(happy_score)
                attractive_scores.append(attr_score/len(faces))
                face_blur_scores.append(blur_score/len(faces))
            else:
                continue
        else:
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                model_input = np.expand_dims(face_roi, axis=0)
                img_features = extract_features(face_roi)
                prediction = model_emotion.predict(img_features)
                outputlabel = labels[np.argmax(prediction)]
                print("Emotion: ", outputlabel)
                if outputlabel == 'happy':
                    happy_score = prediction[0][3]
                    happy_frames.append(frame)
                    print("Happy Score: ", happy_score)
                    happy_scores.append(happy_score)
                    img_attrac = preprocess_image_for_attractive(face_roi)
                    attractive_prediction = attractive_model.predict(img_attrac)
                    attractive_scores.append(attractive_prediction[0][0])
                    face_blur_scores.append(attractive_prediction[0][1])

    blur_scores = []
    clarity_scores = []
    exposure_scores = []

    for frame in happy_frames:
        blur_score = get_blurrness_score(frame)
        clarity_score = calculate_clarity(frame)
        exposure_score = calculate_exposure(frame)
        exposure_scores.append(exposure_score)
        clarity_scores.append(clarity_score)
        blur_scores.append(blur_score)

    frame_zip = zip(happy_frames,attractive_scores,face_blur_scores,blur_scores, clarity_scores, exposure_scores, happy_scores)
    frame_zip = sorted(frame_zip, key=lambda x: (x[2],-x[1], x[3], -x[4]))

    happy_frames = [x[0] for x in frame_zip]
    # happyframes = top 10 frames with highest happy scores
    happy_zip = zip(happy_frames, happy_scores)
    happy_zip = sorted(happy_zip, key=lambda x: -x[1])
    happy_frames = [x[0] for x in happy_zip[:20]]


    print("Frames with highest happy scores: ", len(happy_frames))

    for frame in happy_frames:
        img_attrac = preprocess_image_for_attractive(frame)
        attractive_prediction = attractive_model.predict(img_attrac)
        attractive_scores.append(attractive_prediction[0][0])

    # get top 10 frames with highest attractive scores
    frame_zip = zip(happy_frames, attractive_scores)
    frame_zip = sorted(frame_zip, key=lambda x: -x[1])
    happy_frames = [x[0] for x in frame_zip[:5]]
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in happy_frames]

    return rgb_frames

interface = gr.Blocks()

with interface:
    gr.Markdown("# Best Video Frame Extraction")

    gr.Markdown("Upload a video file to process and extract the best frames.")
    video_input = gr.Video()
    output_images = gr.Gallery(label="Processed Frames")
    process_button = gr.Button("Process Video")
    process_button.click(fn=process_video, inputs=video_input, outputs=output_images)

interface.launch()
