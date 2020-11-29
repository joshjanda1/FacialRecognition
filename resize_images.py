# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:57:33 2020

@author: Josh
"""

import cv2
import numpy as np
import pandas as pd
import os

train_dir = 'F:/GATech/FA20/ISYE6740/Project/train_images/'
test_dir = 'F:/GATech/FA20/ISYE6740/Project/test_images/'
train_resized = 'F:/GATech/FA20/ISYE6740/Project/train_resized/'
test_resized = 'F:/GATech/FA20/ISYE6740/Project/test_resized/'


ytrain = pd.read_csv('train_features.csv', header = None)
ytest = pd.read_csv('test_features.csv', header = None)
#resize images

def resize_images(image_dir, features, save_dir, shape = (288, 288)):
    
    all_images = os.listdir(image_dir)
    resized_df = pd.DataFrame()
    
    for img_name in all_images:
        
        img = cv2.imread(image_dir + img_name)
        img = cv2.resize(img, dsize = shape, interpolation = cv2.INTER_CUBIC)
        img_features = features[features[0] == image_dir + img_name]
        scale = shape[0] // 96
        
        for feature in img_features.iterrows():
        
            filepath, x1, y1, x2, y2, class_name = feature[1]
            x1, y1, x2, y2 = int(x1*3), int(y1*3), int(x2*3), int(y2*3)
            filepath = save_dir + img_name
            
            resized_df = resized_df.append([[filepath, x1, y1, x2, y2, class_name]])
        
        cv2.imwrite(save_dir + img_name, img)
    
    return resized_df

ytrain_resized = resize_images(train_dir, ytrain, train_resized)
ytest_resized = resize_images(test_dir, ytest, test_resized)
                  
            
ytrain_resized.to_csv('ytrain_resized.csv', index = False, header = False)          
ytest_resized.to_csv('ytest_resized.csv', index = False, header = False)     

for img_name in all_images:
    
    img = cv2.imread(train_resized + img_name)
    all_images2.append(img)
    