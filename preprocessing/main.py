# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 07:08:21 2022

@author: yhosni18
"""
import video_sampling_labeling
import argparse

        

# Create the parser
parser = argparse.ArgumentParser(description='Different preprocessing steps')

parser.add_argument('--method', type=str, required=True, help='Define the preprocessing step')
parser.add_argument('--videoPath', type=str, required=True, help='The main path to the vidoes')
parser.add_argument('--videoList', type=str, required=True, help='The path to the vidoe list with the vidoe names')
parser.add_argument('--savingPath', type=str, required=True, help=' The saving path to the vidoes and labels')


args = parser.parse_args()


if args.m == 'video_sampling_labeling':
    video_sampling_labeling.resampling_labeling(args.videoList, args.videoPath, args.savingPath)
else:
    raise ValueError('Error: Unknow preprocessing step')    
    
