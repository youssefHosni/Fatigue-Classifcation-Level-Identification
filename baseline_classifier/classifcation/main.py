# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 18:27:59 2022

@author: yhosni18
"""

import pandas as pd
import numpy as np
import os 
import cv2

from pretrained_models import Inception_v3
from pretrained_models import VGG_16
from pretrained_models import simple_CNN
from pretrained_models import MobileNet
from pretrained_models import InceptionResNetV2
import train 
from sklearn.model_selection import train_test_split

def select_CNN_model(model_name,num_classes,is_trainable,input_shape):
    
    if model_name == 'simple_CNN':
        model= simple_CNN(input_shape,num_classes)
    
    elif model_name=='MobileNet':
        model=MobileNet(num_classes,is_trainable)
    
    elif model_name=='VGG-16':
        model=VGG_16(num_classes,is_trainable)
    
    elif model_name=='Inception-v3':
        model=Inception_v3(num_classes,is_trainable)
    
    elif model_name=='InceptionResNetV2':
        model=InceptionResNetV2(num_classes,is_trainable)
        
    else:
        print("Error value : There is no model with the following name",model_name)
        return 
    
    return model
        
def getLayerIndexByName(model, layername):
    
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
    return None    
        
    
def load_images(folder,width,height):
    images = []
    i=0
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img=cv2.resize(img,(width,height))
        if img is not None:
            images.append(img)
            i=i+1
    return images



main_path = '/home/youssef/Phd_Work/Fatigue_Classification/baseline_classifier/Data'
model_saving_patht = 'D:\Fatigue_classification\Baseline_classifier\classifcation\saved_models'
width = 224
height = 224

train_data = load_images(os.path.join(main_path,'Train_Data_test'), width, height)
test_data = load_images(os.path.join(main_path,'Test_Data_test'), width, height)
train_labels = pd.read_csv(os.path.join(main_path,'train_labels_test.csv'))
train_labels = train_labels['label']
test_labels = pd.read_csv(os.path.join(main_path,'test_labels_test.csv'))
test_labels = test_labels['label']
train_data, val_data, train_labels, val_labels = train_test_split(train_data,train_labels, test_size=0.33)

train_data = np.array(train_data)
val_data = np.array(val_data)
test_data = np.array(test_data)

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)


train_data = train_data.astype('float32')
val_data = val_data.astype('float32')
test_data = test_data.astype('float32')

train_labels = train_labels.astype('float32')
val_labels = val_labels.astype('float32')
test_labels = test_labels.astype('float32')



# model configuration 

model_name = 'MobileNet'
num_classes = 2
is_trainable = False
input_shape = (width, height)
model = select_CNN_model(model_name,num_classes,is_trainable,input_shape)
num_epoch = 10
batch_size = 64
evaulation_metric = 'accuracy'

predicted_labels = train.training_model(model, train_data, train_labels, val_data, val_labels, test_data, test_labels, num_epoch, batch_size, num_classes, evaulation_metric)
