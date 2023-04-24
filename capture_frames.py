import cv2
import numpy as np
import os
import shutil
import glob
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from main import _MAIN_PATH, DATA_PATH, content_directory, video_path, details, info_csv 
from matplotlib import pyplot as plt


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

loc = video_labelled['video location'].values

videos_tot_frames = []
videos_fps = []
for url in loc:
    vid = cv2.VideoCapture(str(url))

    tot_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    per_sec = vid.get(cv2.CAP_PROP_FPS)

    videos_tot_frames.append(tot_frames)
    videos_fps.append(per_sec)

## import frames for words labelled

word_frame = pd.read_excel(os.path.join(DATA_PATH,'corpus_csv_files','ISL_CSLRT_Corpus_word_details.xlsx'))

frame_data = word_frame.copy()

frame_data["Frames path"] =  _MAIN_PATH +'/'+ frame_data["Frames path"]



image_gen = ImageDataGenerator(rescale=1. / 255)

imds = image_gen.flow_from_dataframe(dataframe = frame_data, 
x_col = "Frames path",
y_col = "Word",
target_size=(224,224),color_mode='rgb')


[im , lab] = imds.next()
imds.classes

for i in range(1,26):
    plt.subplot(5, 5, i)

    plt.imshow(im[i])
    plt.title([j for j in imds.class_indices if imds.class_indices[j]==np.argmax(lab[i],axis=0)])
plt.show(block=False)
