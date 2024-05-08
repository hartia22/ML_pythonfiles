# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:52:37 2021

@author: Majic
"""

import numpy as np
import os
import imageio
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from skimage.transform import resize as imresize
from tqdm import tqdm


img_main = np.load('\images.npy')