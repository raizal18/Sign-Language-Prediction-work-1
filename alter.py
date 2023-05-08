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
import tensorflow_hub as hub
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

enc = np.load('encoder.npy', allow_pickle=True)[0]

inverse_transformer = {value:key for key,value in enc.items()}


# Encode Train Labels

 

try:

    feature_extractor = load_model('trained_weights/feature_extractor.h5')

except FileNotFoundError:

    raise FileNotFoundError


vid_feat = []

for idx, i in enumerate(range(video_labelled.shape[0])):

    vid_feat.append(np.reshape(np.load(f"features/{format(idx,'03d')}.npy"),(392,256)))

vid_feat = np.array(vid_feat)

x_train, x_test, y_train, y_test = train_test_split(vid_feat, train_df['tag'].values, test_size=0.3)



model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))







