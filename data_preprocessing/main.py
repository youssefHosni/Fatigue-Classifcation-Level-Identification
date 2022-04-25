# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:54:21 2022

@author: yhosni18
"""

import video_sampling
import argparse

# Create the parser
parser = argparse.ArgumentParser()

parser.add_argument('--m', type=str, required=True)
parser.add_argument('--i', type=str, required=True)
parser.add_argument('--o', type=str, required=True)
parser.add_argument('--l', type=str)
parser.add_argument('--t', type=int)

args = parser.parse_args()


if args.m == 'video_sampling':
    video_sampling.video_sampling(args.i, args.o, args.l, args.t)
    
    