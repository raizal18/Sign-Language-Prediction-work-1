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
from keras import layers
import mediapipe as mp


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


# ## import frames for words labelled

word_frame = pd.read_excel(os.path.join(DATA_PATH,'corpus_csv_files','ISL_CSLRT_Corpus_word_details.xlsx'))

frame_data = word_frame.copy()

frame_data["Frames path"] =  _MAIN_PATH +'/'+ frame_data["Frames path"]

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

def get_feature(sample,sample_label):
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities


    def mediapipe_detection(image, model):
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results


 
    cap = cv2.VideoCapture(sample)

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            if not ret:

                print("Can't receive frame (stream end?). Exiting ...")
                cap.release()
                cv2.destroyAllWindows()
                break
            # Make detections
            frame = cv2.resize(frame, (720, 480),interpolation = cv2.INTER_LINEAR)
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            # Show to screen

            org = (5, 40)
            
            # fontScale
            fontScale = 1
            
            # Blue color in BGR
            color = (38, 249, 255)
            
            # Line thickness of 2 px
            thickness = 1
            
            # Using cv2.putText() method
            font = cv2.FONT_HERSHEY_COMPLEX
            image = cv2.putText(image,f"Label: {sample_label}", org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
    
            cv2.imshow('Joint Position & action ', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



sample = video_labelled['video location'].values[17]
sample_label = video_labelled['label'].values[17]

get_feature(sample,sample_label)

EXTRACT_FEATURE_VIDEO =False
def mediapipe_detection(image, model):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def get_results(sample):

    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities    
    cap = cv2.VideoCapture(sample)
    cap.set(cv2.CAP_PROP_FPS, 10)
    feat_list = []
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            try :
                frame = cv2.resize(frame, (720, 480),interpolation = cv2.INTER_LINEAR)
                image, results = mediapipe_detection(frame, holistic)
                feat_list.append(results)
            except:
                cap.release()
                return np.array([extract_keypoints(result) for result in feat_list])

def get_frame_feature(sample):
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    frame = cv2.imread(sample)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        try :
            frame = cv2.resize(frame, (720, 480),interpolation = cv2.INTER_LINEAR)
            image, results = mediapipe_detection(frame, holistic)
            return results
        except:
            return np.zeros((1,1662))



frame_meta = []
frame_label = []
for idx,(word, frame) in enumerate(zip(frame_data["Word"],frame_data["Frames path"])):
    if idx%10 == 0:
        print(f"processed frames {idx} remaining {frame_data.shape[0]-idx}")
    frame_meta.append(extract_keypoints(get_frame_feature(frame)))
    frame_label.append(word)

frame_meta = np.array(frame_meta)
    
    



if EXTRACT_FEATURE_VIDEO == True:

    lab_det = []
    feat_fname = []
    idx = 0
    for sample,label in zip(video_labelled['video location'],video_labelled['label']):
        
        lab_det.append(label)
        feat_fname.append(os.path.join('features',f"{idx}.npy"))
        print(f"file {sample} in progress")
        np.save(os.path.join('features',f"{idx}.npy"),get_results(sample))
        idx += 1
        # np.save(os.path.join('features',f"{idx}.npy"),feat)
    pd.DataFrame(np.array([lab_det,feat_fname]).transpose(),columns=['label','stored feature']).to_csv('feature_details.csv')

FEATURE_PATH =  'features'  


feature_data = pd.read_csv('feature_details.csv')

label_video = feature_data['label']
meta_video= feature_data['stored feature']

meta_data = []
for file in meta_video:
    meta_data.append(np.load(file))

shp = [da.shape[0] for da in meta_data]




meta_data = np.array(meta_data)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

enc =  LabelEncoder()
word_enc = LabelEncoder()
y_int =enc.fit_transform(label_video)
y_int_frame = word_enc.fit_transform(frame_label)

y = to_categorical(y_int)
y_frame = to_categorical(y_int_frame)

fdata = np.expand_dims(frame_meta,axis=1)

X_train, X_test, y_train, y_test = train_test_split(fdata, y_frame, test_size=0.4)


from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)



model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(1,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

MODEL_RETRAIN = False

if MODEL_RETRAIN == True:
    model.fit(fdata, y_frame, epochs=1000, callbacks=[tb_callback])
else:
    model = tf.keras.models.load_model('model.h5')

sentence = []
print('Testing Models in sentence data')
for idx, test in enumerate(meta_data[1:276]):
    sentence.append(' '.join(list(word_enc.inverse_transform(np.argmax(model.predict(np.expand_dims(test, axis=1)),axis=1)))))

from confusion import confusion

met = confusion(np.array(label_video[1:276]), np.array(sentence))

cm = met.getmatrix()

[acc, pre, rec, fsc] = met.metrics()

if not os.path.exists('model.h5'):
    model.save('model.h5')


def read_test_data(sample, model, word_enc):
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
                    # print(extract_keypoints(results))
                    pred = word_enc.inverse_transform(np.argmax(model.predict(np.expand_dims(np.expand_dims(extract_keypoints(results),axis=0), axis=1)),axis=1))
                    draw_styled_landmarks(image,results)
                    org = (5, 40)
                    fontScale = 1
                    color = (38, 249, 255)
                    thickness = 1
                    font = cv2.FONT_HERSHEY_COMPLEX
                    image = cv2.putText(image,f"sign: {pred}", org, font, 
                                    fontScale, color, thickness, cv2.LINE_AA)
    
                    cv2.imshow('test data', image)
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                    
            else:    
                    cap.release()
                    cv2.destroyAllWindows()
                     
                    break


SHOW_VIDEOS = False
if SHOW_VIDEOS == True:
    n = 120

    read_test_data(video_labelled['video location'][n], model, word_enc)

    print(f"original {video_labelled['label'][n]}")
