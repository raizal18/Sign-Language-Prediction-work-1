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

def format_frames(frame, output_size):

  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """

  result = []
  src = cv2.VideoCapture(str(video_path))  

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)

  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result


sample_video = frames_from_video_file(video_labelled['video location'][2],output_size=(224, 224),n_frames=5,frame_step=8)

for im in sample_video:
    cv2.imshow('frame', im)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

hub_url = "movinet_a2/3"

encoder = hub.KerasLayer("movinet_a2/3", trainable=True)

inputs = tf.keras.layers.Input(
    shape=[None, None, None, 3],
    dtype=tf.float32,
    name='image')

# [batch_size, 600]

outputs = encoder(dict(image=inputs))

model = tf.keras.Model(inputs, outputs, name='movinet')
capture_action = False

if capture_action == True:

    print(f"start time {time.asctime()}")

    for idx, (file, label) in enumerate(zip(video_labelled['video location'],video_labelled['label'])):

        # if (idx > 53) & (idx % 2 == 0):        
        sample_video = frames_from_video_file(file,output_size=(224, 224),n_frames=5,frame_step=8)
        example_output = model(np.expand_dims(sample_video,axis=0))
        np.save(f"action/{idx}.npy", example_output)
        print(f"{idx} time taken {time.asctime()} ")



print(example_output)