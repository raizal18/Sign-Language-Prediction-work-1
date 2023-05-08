import os
import cv2
import time
import pickle
import gradio as gr
import numpy as np
import pandas as pd
from keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf
from numpy import random
from sklearn.preprocessing import LabelEncoder
from main import _MAIN_PATH, DATA_PATH, content_directory, video_path, details, info_csv 
from tensorflow.keras.applications.resnet50 import preprocess_input
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

enc = np.load('encoder.npy', allow_pickle=1)[1]

norm = lambda loc : '/'.join(['/'.join(i.split('\\')) for i in loc.split('/')])

def create_subtitles(words, video_duration, filename):

    word_duration = video_duration / len(words)
    start_time = 0
    end_time = word_duration
    subtitles = []
    for i, word in enumerate(words):
        subtitle = f"{i+1}\n{word}\n"
        subtitle += f"{format_time(start_time)} --> {format_time(end_time)}\n"
        subtitles.append(subtitle)
        start_time = end_time
        end_time += word_duration

    with open(filename, "w") as f:
        f.write("".join(subtitles))


def format_time(time):
    hours = int(time / 3600)
    minutes = int((time % 3600) / 60)
    seconds = int(time % 60)
    milliseconds = int((time - int(time)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


# [batch_size, 600]

model = load_model('feature_extractor.h5')
sign_predictor = load_model('sign_predictor_weights.h5')

# def _action_extract(video_path):

#     print(f"start time {time.asctime()}")
#     # if (idx > 53) & (idx % 2 == 0):        
#     sample_video = frames_from_video_file(video_path, output_size=(224, 224),n_frames=5,frame_step=8)
#     example_output = model(np.expand_dims(sample_video,axis=0))
#     print(f"time taken {time.asctime()} ")
#     return example_output

# def video_demo(video):
#     video = norm(video)
#     global pre, indx
#     pre = False
#     for num, loc in enumerate(video_labelled['video location']):
#         if norm(loc).split('/')[-1] == video.split('/')[-1]:
#             pre = True
#             indx = num
#     if pre :
#         prob = sign_predictor.predict(np.load(f"action/{num}.npy"))
#         text = enc[np.argmax(prob, axis=1)[0]]
#     else :
#         prob = sign_predictor.predict(_action_extract(video))
#         text = enc[np.argmax(prob, axis=1)[0]]
#     return video, text

def _get_test_feat(path:str):
    for idx, path in enumerate(path):
        cap = cv2.VideoCapture(path)
        frame_rate = 1
        features = []
        maxf = 1
        while cap.isOpened():
            ret, frame = cap.read()
            maxf += 1
            if maxf >100:
                break
            if not ret:
                break

            resized_frame = cv2.resize(frame, (224, 224))

            preprocessed_frame = preprocess_input(resized_frame)

            features.append(model.predict(np.array([preprocessed_frame]),verbose=0))

            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_rate)

        features = np.concatenate(features, axis=0)
        
        da = [fm.flatten() for fm in features]
        arr_list_2d = [arr.reshape(1, -1) for arr in da]
        return np.mean(np.squeeze(np.array(arr_list_2d)), axis=0)



def prediction_confident(path:str):

    feat = np.expand_dims(np.reshape(_get_test_feat(path),(392,256)),axis=-1)
    prob = sign_predictor(feat)
    label_id =  np.argmax(prob,axis=0)
    label = enc[label_id]
    return label


demo = gr.Interface(
    fn=prediction_confident,
    inputs=[
        gr.Video(type="file", label="In", interactive=True),
     ],
    outputs=[gr.Video(label="Out"),
             gr.Label(label='sign')
    ],
    examples=[
        video_labelled['video location'][2],
        video_labelled['video location'][50],
        video_labelled['video location'][120],
    ],
)



if __name__ == "__main__":
    demo.launch()

