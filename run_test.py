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

with open('encode.pkl', 'rb') as f:
    enc = pickle.load(f) 



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

def format_frames(frame, output_size):

  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
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


encoder = hub.KerasLayer("movinet_a2/3", trainable=True)

inputs = tf.keras.layers.Input(
    shape=[None, None, None, 3],
    dtype=tf.float32,
    name='image')

# [batch_size, 600]

outputs = encoder(dict(image=inputs))

model = tf.keras.Model(inputs, outputs, name='movinet')


sign_predictor = load_model('sign_predictor_weights.h5')

def _action_extract(video_path):

    print(f"start time {time.asctime()}")
    # if (idx > 53) & (idx % 2 == 0):        
    sample_video = frames_from_video_file(video_path, output_size=(224, 224),n_frames=5,frame_step=8)
    example_output = model(np.expand_dims(sample_video,axis=0))
    print(f"time taken {time.asctime()} ")
    return example_output

def video_demo(video):

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    prob = sign_predictor.predict(_action_extract(video))
    text = enc.inverse_transform(np.argmax(prob, axis=1))
    sub = create_subtitles(text[0].split(' '), duration, 'files/output.srt')
    s3 = os.path.join(os.path.abspath(''), "files/output.srt")
    return [video,s3], text[0]



demo = gr.Interface(
    fn=video_demo,
    inputs=[
        gr.Video(type="file", label="In", interactive=True),
     ],
    outputs=[gr.Video(label="Out"),
             gr.Label(label='sign')
    ],
    examples=[
        video_labelled['video location'][2],
        video_labelled['video location'][3],
        video_labelled['video location'][5],
    ],
)

if __name__ == "__main__":
    demo.launch()