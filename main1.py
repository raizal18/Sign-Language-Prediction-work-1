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
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Softmax, ReLU, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import layers
from keras.models import load_model
import mediapipe as mp
import logging
from temporal_gcnn import model
from confusion import confusion
#import  Speech Engine 

import pyttsx3

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
def draw_styled_landmarks(image, results):
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                            ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                            ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                            ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            ) 


play_sample = False

if play_sample == True:
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

def mediapipe_detection(image, model):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def read_test_data(sample):
    mp_holistic = mp.solutions.holistic 
    mp_drawing = mp.solutions.drawing_utils 

    cap = cv2.VideoCapture(sample)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = round(fps / 24)
    list_frm = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:

                if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_interval == 0:
                    frame = cv2.resize(frame, (720, 480),interpolation = cv2.INTER_LINEAR)
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    cv2.imshow('test data', image)
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                    
            else:    
                    cap.release()
                    cv2.destroyAllWindows()
                     
                    break

def _action_extract(video_path):

    print(f"start time {time.asctime()}")
    # if (idx > 53) & (idx % 2 == 0):        
    sample_video = frames_from_video_file(video_path, output_size=(224, 224),n_frames=5,frame_step=8)
    example_output = model(np.expand_dims(sample_video,axis=0))
    print(f"time taken {time.asctime()} ")
    return example_output


action_data = [] 
for i in range(0,687):
    tmp = np.load(f"action/{i}.npy")
    action_data.append(tmp)

action_data = np.array(action_data)

lab = video_labelled['label'].values

uni, counts = np.unique(lab, return_counts = True)


rm_values = uni[counts<7]

X = []
Y = []
for feature, label in zip(action_data, lab):
    if label not in rm_values:
        X.append(feature)
        Y.append(label)
X = np.squeeze(np.array(X))
Y = np.array(Y)

unique_val = np.unique(Y)
class_indices = {k:v for v,k in enumerate(unique_val)}

inverse_key = {k:v for k,v in enumerate(unique_val)}

np.save('encoder.npy', [class_indices, inverse_key])
Y_int = np.array([class_indices[clss] for clss in Y])


from keras.utils import to_categorical

Y_one = to_categorical(Y_int)

x_train, x_test, y_train, y_test = train_test_split(X,Y_one, test_size = 0.40)

inputs = Input(shape=(600,))
# conv1 = Conv1D(100,6)
# relu1 = ReLU()(conv1)
# conv2 = Conv1D(100,6)(relu1)
dens1 = Dense(300, activation = 'relu')(inputs)
dens2 = Dense(100, activation = 'relu')(dens1)
smax = Dense(Y_one.shape[1], activation ='softmax')(dens2)

mod = Model(inputs = inputs, outputs = smax)

mod.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy(),
                       tf.keras.metrics.FalseNegatives()])

hist = mod.fit(X, Y_one, epochs=100, verbose=1)

mod.evaluate(x_test, y_test)



for i in range(80,120):
  test_video = video_labelled['video location'][i]
  test_label = video_labelled['label'][i]

  action = _action_extract(test_video)

  test_pred = mod.predict(action)

  pred_label = inverse_key[np.argmax(test_pred,axis=1)[0]]

  print(f"{test_label} -> {pred_label}")
# text = list(enc.inverse_transform(np.argmax(test_pred, axis=1)))

def show_sign_prediction(sample, ):
    mp_holistic = mp.solutions.holistic 
    cap = cv2.VideoCapture(sample)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = round(fps / 24)
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:

            if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_interval == 0:
                frame = cv2.resize(frame, (720, 480),interpolation = cv2.INTER_LINEAR)
                cv2.imshow('test data', frame)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                
        else:    
                cap.release()
                cv2.destroyAllWindows()
                  
                break

