# provides access to system-specific parameters and functions
import sys
# helps with file and directory operations, enviornment variables, and process control
import os
# for image loading, processing, and manipulation
import cv2
import shutil

import matplotlib # for plotting and visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from mpl_toolkits.mplot3d import axes3d # for creating 3D plots
from tqdm import tqdm # library used for displaying progress bars for loops
from sklearn.model_selection import train_test_split

import tensorflow as tf

import pathlib

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random 

data_dir = pathlib.Path(DATASET_PATH)

IMG_SIZE = 176 # resize images to 176 x 176
BRIGHT_RANGE = [0.8, 1.2] # adjust brightness to 80 - 120% of original
ZOOM = [0.99, 1.01] # range of zoom -- 1% in or out
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"

# imagedatagenerators
# rescale: normalizes pixel values by dividing by 255 --> bring values in range [0, 1]
# brightness_range: randomly adjusts brightness within specified range 
# zoom_range: randomly zoomes in/out within specifed range
# horizontal_flip: randomly flips images horizontally for more variety
# want augmentation in training data but not test/validation --> just rescale
train_image_generator = ImageDataGenerator(rescale = 1./255, brightness_range = BRIGHT_RANGE, zoom_range = ZOOM, data_format = DATA_FORMAT, fill_mode = FILL_MODE, horizontal_flip = True)
