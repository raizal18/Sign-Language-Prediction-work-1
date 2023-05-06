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
from keras.preprocessing.image import ImageDataGenerator
from main import _MAIN_PATH, DATA_PATH, content_directory, video_path, details, info_csv 
from keras import layers
import mediapipe as mp
import logging
from keras.utils import Progbar

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










