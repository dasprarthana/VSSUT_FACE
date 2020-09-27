# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:38:58 2020

@author: 91985
"""

import os
import cv2
import numpy as np
import pickle

data_dir = os.path.join(os.getcwd(),'clean_data')

img_dir = os.path.join(os.getcwd(),'Images')


def preprocess(image):
    image = cv2.resize(image,(100,100))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return image


Images = []
labels = []


for i in os.listdir(img_dir):
    image = cv2.imread(os.path.join(img_dir,i))
    image = preprocess(image)
    Images.append(image)
    labels.append(i.split("_"))


Images = np.array(image)
labels = np.array(labels)

with open(os.path.join(data_dir,'Images.p'),'wb') as f:
    pickle.dump(Images,f)

with open(os.path.join(data_dir,'labels.p'),'wb') as f:
    pickle.dump(labels,f)
