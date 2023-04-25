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

# loc = video_labelled['video location'].values

# videos_tot_frames = []
# videos_fps = []
# for url in loc:
#     vid = cv2.VideoCapture(str(url))

#     tot_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

#     per_sec = vid.get(cv2.CAP_PROP_FPS)

#     videos_tot_frames.append(tot_frames)
#     videos_fps.append(per_sec)

# ## import frames for words labelled

# word_frame = pd.read_excel(os.path.join(DATA_PATH,'corpus_csv_files','ISL_CSLRT_Corpus_word_details.xlsx'))

# frame_data = word_frame.copy()

# frame_data["Frames path"] =  _MAIN_PATH +'/'+ frame_data["Frames path"]



# image_gen = ImageDataGenerator(rescale=1. / 255)

# imds = image_gen.flow_from_dataframe(dataframe = frame_data, 
# x_col = "Frames path",
# y_col = "Word",
# target_size = (224, 224), color_mode='rgb')


# # [im, lab] = imds.next()


# # for i in range(1,26):
# #     plt.subplot(5, 5, i)

# #     plt.imshow(im[i])
# #     plt.title([j for j in imds.class_indices if imds.class_indices[j]==np.argmax(lab[i],axis=0)])
# # plt.show(block=False)



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


    def draw_styled_landmarks(image, results):
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

EXTRACT_FEATURE =False
if EXTRACT_FEATURE == True:
    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def get_results(sample):
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
                    return np.array([extract_keypoints(results) for result in feat_list])
                    



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


