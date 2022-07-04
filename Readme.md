# Fatigue Classification and Level Identification 

In this project, thermal images were used to classify people in fatigue from people in rest and to identify the fatigue level of them. This project is part of my phd work at VisualHealth team at the center of machine vision and signal analysis (CMVS) at university of Oulu.

The following pipeline is followed (still in progress):
* Image preprocesing 
* Landmark labeling using Imaglab
* Face alignment using MTCNN 
* Baseline classfication using pretrained based CNN models such as Mobilenet, Resnet, VGG
* Advanced vidoe classifcation using MTV (Multiview transfomrers for video classifcation)
