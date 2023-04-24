import os
import sys
import shutil
import glob
import logging
import pandas as pd
import numpy as np
from read_video import format_frames, frames_from_video_file
import matplotlib.pyplot as plt
import cv2
import imageio
import math
from tensorflow_docs.vis import embed
logging.basicConfig(filename='logg.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This message will get logged on to a file')

_MAIN_PATH = DATA_PATH = "F:/ISL_CSLRT_Corpus/"
DATA_PATH = "F:/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus"

content_directory = os.listdir(DATA_PATH)

info_csv = os.listdir(os.path.join(DATA_PATH, content_directory[0]))

details = pd.read_excel(os.path.join(DATA_PATH, content_directory[0], info_csv[1]))


video_path = os.path.join(_MAIN_PATH, details['File location'].values[1])

from framer import frames_from_video_file

sample_video = frames_from_video_file(video_path, n_frames = 24)
sample_video.shape
