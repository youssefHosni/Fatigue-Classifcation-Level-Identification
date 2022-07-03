# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 18:29:42 2022

@author: yhosni18
"""

import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def f1_micro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp_per_class = K.sum(K.cast(y_true*y_pred, 'float'), axis=1)
    tn_per_class = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=1)
    fp_per_class = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=1)
    fn_per_class = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=1)

    p_per_class = tp_per_class / (tp_per_class + fp_per_class + K.epsilon())
    r_per_class = tp_per_class / (tp_per_class + fn_per_class + K.epsilon())

    f1_per_class = 2*p_per_class*r_per_class / (p_per_class+r_per_class+K.epsilon())
    f1_total= K.sum(f1_per_class*K.sum(y_true,axis=1))/ K.sum(y_true)
    
    return f1_total



def confusion_matrix_calc(predicted_labels,true_labels,num_classes,title):
    positions = np.arange(0,num_classes)
    classes = np.arange(0,num_classes)
    predicted_labels 
    cm = confusion_matrix(predicted_labels.argmax(axis=1), true_labels, labels=(0,1)) 
    disp = ConfusionMatrixDisplay(cm,display_labels=classes)
    plt.figure(figsize=(10,10))
    disp.plot()
    plt.xticks(positions, ['Rest','Fatigue'])
    plt.yticks(positions, ['Rest','Fatigue'])
    plt.title(title)
    return 
    

