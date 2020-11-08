# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:26:16 2020

@author: Josh
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import ast
import cv2

train_data = pd.read_csv('training.csv')
train_data.fillna(method = 'ffill', inplace = True)
#expand Image column into 96*96 vector
train_images = train_data['Image'].str.split(' ', 96*96, expand = True)
train_features = train_data.drop('Image', axis = 1)

for i in np.random.randint(low = 0, high = 7048, size = 10):
    
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 12))
    image = train_images.iloc[i, :].to_numpy(dtype = np.float64)
    image_copy = image.copy().reshape(96, 96)
    features = train_features.iloc[i, :]
    axes[0].imshow(image.reshape(96, 96))
    
    feature_dict = {'leye' : [features['left_eye_inner_corner_x'], features['left_eye_inner_corner_y'],
                              features['left_eye_outer_corner_x'], features['left_eye_outer_corner_y'],],
                    'reye' : [features['right_eye_inner_corner_x'], features['right_eye_inner_corner_y'],
                              features['right_eye_outer_corner_x'], features['right_eye_outer_corner_y']],
                    'leyebrow' : [features['left_eyebrow_inner_end_x'], features['left_eyebrow_inner_end_y'],
                              features['left_eyebrow_outer_end_x'], features['left_eyebrow_outer_end_y']],
                    'reyebrow' : [features['right_eyebrow_inner_end_x'], features['right_eyebrow_inner_end_y'],
                              features['right_eyebrow_outer_end_x'], features['right_eyebrow_outer_end_y']],
                    'mouth' : [features['mouth_left_corner_x'], features['mouth_left_corner_y'],
                              features['mouth_right_corner_x'], features['mouth_right_corner_y']],
                    'nose' : [features['nose_tip_x'], features['nose_tip_y']]
                    }
    
    for feature, feature_loc in feature_dict.items():
        
        feature_loc = np.array(feature_loc, dtype = np.int64)
        
        if feature == 'nose':
            
            x, y, xmax, ymax = feature_loc[0] - 8, feature_loc[1] - 18, feature_loc[0] + 8, feature_loc[1] + 8
        
        else:
   
            x, y, xmax, ymax = feature_loc[0], feature_loc[1] - 2, feature_loc[2], feature_loc[3] + 2
        
        cv2.rectangle(image_copy, (x, y), (xmax, ymax), (255, 0, 0), 1)
        cv2.putText(image_copy, feature, (xmax, y), cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 0, 0), 1)
        
    axes[1].imshow(image_copy)

def generate_labels(images, features):
    
    feature_dict = {}
    for i in range(len(images)):
        #get image data
        img = images.iloc[i, :].to_numpy(dtype = np.float64)
        #get feature data
        ftrs = features.iloc[i, :]
        # features are lists in form of x, y, xmax, ymax
        feature_dict['img_{0}'.format(i)] = {
                            'leye' : {'x' : ftrs['left_eye_inner_corner_x'], 'y' : ftrs['left_eye_inner_corner_y'] - 2,
                                      'xmax' : ftrs['left_eye_outer_corner_x'], 'ymax' : ftrs['left_eye_outer_corner_y'] + 2},
                            'reye' : {'x' : ftrs['right_eye_inner_corner_x'], 'y' : ftrs['right_eye_inner_corner_y'] - 2,
                                      'xmax' : ftrs['right_eye_outer_corner_x'], 'ymax' : ftrs['right_eye_outer_corner_y'] + 2},
                            'leyebrow' : {'x' : ftrs['left_eyebrow_inner_end_x'], 'y' : ftrs['left_eyebrow_inner_end_y'] - 2,
                                      'xmax' : ftrs['left_eyebrow_outer_end_x'], 'ymax' : ftrs['left_eyebrow_outer_end_y'] + 2},
                            'reyebrow' : {'x' : ftrs['right_eyebrow_inner_end_x'], 'y' : ftrs['right_eyebrow_inner_end_y'] - 2,
                                      'xmax' : ftrs['right_eyebrow_outer_end_x'], 'ymax' : ftrs['right_eyebrow_outer_end_y'] + 2},
                            'mouth' : {'x' : ftrs['mouth_left_corner_x'], 'y' : ftrs['mouth_left_corner_y'] - 2,
                                      'xmax' : ftrs['mouth_right_corner_x'], 'ymax' : ftrs['mouth_right_corner_y'] + 2},
                            'nose' : {'x' : ftrs['nose_tip_x'] - 8, 'y' : ftrs['nose_tip_y'] - 18,
                                      'xmax' : ftrs['nose_tip_x'] + 8, 'ymax' : ftrs['nose_tip_y'] + 8}
                            }
    #convert dictionary to df and orient by index of img_id, feature
    feature_df = pd.DataFrame.from_dict({(i,j): feature_dict[i][j] 
                           for i in feature_dict.keys() 
                           for j in feature_dict[i].keys()},
                       orient = 'index')
    return feature_df
    
feature_df = generate_labels(train_images, train_features)