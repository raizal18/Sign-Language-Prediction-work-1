import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import time
import os 
import PIL
import cv2
import tensorflow as tf
import tqdm
import shutil
import glob
import pandas as pd
from numpy import random
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.model_selection import train_test_split
from main import _MAIN_PATH, DATA_PATH, content_directory, video_path, details, info_csv 
from keras import layers
import mediapipe as mp
import logging
from keras.utils import Progbar
from temporal_gcnn import model

EXTRACT_FEATURE_FROM_SCRATCH  = False

preprocess_input = tf.keras.application.resnet50.preprocess_input

mpl.rcParams.update({
    'font.size': 10,
})

total_files = 0
file_url = []
class_description = []

files_path =os.path.join(DATA_PATH,'Videos_Sentence_Level/')
for root, direct, files in os.walk(files_path):

    total_files += len(files)
    if root != files_path:

        for lab,url in zip([root.split('/')[-1] for i in files],files):
            class_description.append(lab)
            file_url.append(os.path.join(root,url))

video_labelled = pd.DataFrame(list(zip(file_url,class_description)),columns=["video location", "label"])

train_df = video_labelled.copy()

train_df.columns = ['path', 'tag']

max_dims = []

i_bar = Progbar(train_df.shape[0])
for idx,path in enumerate(train_df['path']):
    
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    max_dims.append(fps)
    i_bar.update(idx)

# get encoder 

if EXTRACT_FEATURE_FROM_SCRATCH == True:

    for idx, (path, label) in enumerate(zip(train_df['path'], train_df['tag'])):
        if not idx<0:
            print(idx)
            # Load video
            cap = cv2.VideoCapture(path)

            # Define frame rate
            frame_rate = 1  # read one frame every second

            # Initialize variables
            features = []
            maxf = 1
            # Loop through frames
            while cap.isOpened():
                # Read frame
                ret, frame = cap.read()
                maxf += 1
                # Check if frame was read successfully
                if maxf >100:
                    break
                if not ret:
                    break

                # Resize frame to match input size of InceptionV3 model
                resized_frame = cv2.resize(frame, (224, 224))

                # Preprocess frame to match input format of InceptionV3 model
                preprocessed_frame = preprocess_input(resized_frame)

                # Extract features from frame using InceptionV3 model
                features.append(model.predict(np.array([preprocessed_frame]),verbose=0))

                # Skip frames to match frame rate
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_rate)

            # Aggregate features from all frames
            features = np.concatenate(features, axis=0)
            
            da = [fm.flatten() for fm in features]
            arr_list_2d = [arr.reshape(1, -1) for arr in da]
            np.save(os.path.join('/content/drive/MyDrive/feat' , f"{format(idx,'03d')}.npy"),np.mean(np.squeeze(np.array(arr_list_2d)), axis=0))



# enc = np.load('encoder.npy', allow_pickle=True)[0]

enc = {key:value for value, key in enumerate(np.unique(train_df['tag'].values))}
inverse_transformer = {value:key for key,value in enc.items()}


# Encode Train Labels

 

try:

    feature_extractor = load_model('feature_extractor.h5')

except FileNotFoundError:

    raise FileNotFoundError


vid_feat = []
i_bar = Progbar(video_labelled.shape[0])
for idx, i in enumerate(range(video_labelled.shape[0])):

    vid_feat.append(np.expand_dims(np.reshape(np.load(f"feat/{format(idx,'03d')}.npy"),(392,256)),axis=-1))
    i_bar.update(idx)

vid_feat = np.array(vid_feat)


x_train, x_test, y_train, y_test = train_test_split(vid_feat, train_df['tag'].values, test_size=0.4)

label_int = np.array([enc[inst] for inst in train_df['tag'].values])

label_one = to_categorical(label_int)



batch_size = 5

dataset = tf.data.Dataset.from_tensor_slices((vid_feat,label_one))

dataset = dataset.batch(batch_size)

history = model.fit(dataset, epochs=10)







