# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 07:10:35 2022

@author: yhosni18
"""
import numpy as np
import pandas as pd 
import cv2

def fatigue_level_calculation(frame_name, frame_order ,fps, decaying_rate):
    """ Calculating the fatigue level for eavh frame. The fatigue level range 
    is from 100 to 0. The fatigue level depends on the frame order and the class.
    If the class is resting, then the fatigue level is 0, else the fatigue level
    is calcualted as an exponatial decaying fuction with given decaying rate. If the 
    input frame is after 3 minutes then it will also be zero, because the particpant 
    will be propbably recoverd by then.
    
    Parameters
    ----------
    frame_name : string
        The name of the input frame. This used to know the class of the frame and 
        the frame order
    frame_order: int
        The order of the frame. This is used to calcualte the fatigue level, since 
        as the order incereases the fatigue level decreases
    fps : float 
        Numbe of frame per second for each vidoe.
    decaying_rate : float
        The decaying rate of the expantional function that models the fatigue level

    Returns
    -------
    fatigue level: float
        The calculated fatigue level
            
        
    """
    frame_class = frame_name[0]
    if (frame_class == 'R') or (frame_order >= fps*3*60):
        fatigue_level = 0
    elif frame_class =='F':
        fatigue_level = 100 * np.exp(-frame_order/decaying_rate)
    else:
        raise ValueError('Invalid frame class')
    return fatigue_level

def resampling_labeling(video_list_path, video_main_path, saving_path):
    """ resample the vidoes into single frame and calculate the fatigue level 
    and the label for each frame and save them into a csv file
    Parameters
    ----------
        video_list_path: string 
            The path of the csv file that contains the video names and whether
            it is a train or test video
        
        video_main_path: string 
            The main path of the videos to be loaded. This will be appended to the 
            video name from the video_list_path csv file
        saving_path: string 
            The path to save the frames and the output csv files

    Returns
    -------
    None        
        
    """
    
    video_list = pd.read_csv(video_list_path) # load the csv file with the video names
    # create empty dataframe to save the frame name with it label and fatigue level
    train_df = pd.DataFrame(columns=['image name', 'label', 'fatigue level'])
    test_df = pd.DataFrame(columns=['image name', 'label', 'fatigue level'])
    validation_df = pd.DataFrame(columns=['image name', 'label', 'fatigue level'])
    
    # loop on each video in the input video list
    for index, video_row in video_list.iterrows():
        frame_counter = 0 # intialize the frame counter for each video
        video_path = video_main_path + '/' + video_row['Video name'] + '.mp4'

        # load the video using the given paht and opencv library 
        vid_capture = cv2.VideoCapture(video_path)
        
        if (vid_capture.isOpened() == False):
            print("Error opening the video file")
        else:
            # Read fps and frame count
            # Get frame rate information
            # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
            fps = vid_capture.get(5)
            print(fps)
            # Get frame count
            # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
            frame_count = vid_capture.get(7)
            print(frame_count)
        while(vid_capture.isOpened()):
            # vid_capture.read() methods returns a tuple, first element is a bool 
            # and the second is frame
            ret, frame = vid_capture.read()
            print(video_row['Train/Test'])
            if ret == True:
                # save the frame to the output file
                if video_row['Train/Test'] == 'Train': # check if the video belongs to train or test
                    #save the frame to the train output directory
                    cv2.imwrite((saving_path+'/Train_Data/'+
                        video_row['Video name']+ '_'+
                        str(frame_counter)+'.jpg'), 
                        frame)
                    # calculate the fatigue level for the current frame 
                    decaying_rate = 3*60*fps/10 
                    fatigue_level = fatigue_level_calculation(video_row['Video name'], 
                                                              frame_counter, 
                                                              fps, 
                                                              decaying_rate)
                    # update the train dataframe with the current frame information 
                    train_df = train_df.append({'image name':video_row['Video name']+'_'+str(frame_counter),
                                                'label':video_row['Video name'][0],
                                                'fatigue level':fatigue_level
                                                }, ignore_index=True)
                    # update the frame counter
                    frame_counter = frame_counter + 1
                elif video_row['Train/Test'] == 'Test':
                    # save the image to the test output directory 
                    cv2.imwrite((saving_path+'/Test_Data/'+
                                 video_row['Video name']+'_'+
                                 str(frame_counter)+'.jpg'), 
                                frame)
                    # calculate the fatigue level for the current frame
                    fatigue_level = fatigue_level_calculation(video_row['Video name'], 
                                                              frame_counter, 
                                                              fps, 
                                                              frame_count/10)
                    # update the test data frame with the current frame information 
                    test_df = test_df.append({'image name':video_row['Video name']+'_'+str(frame_counter),
                                                'label':video_row['Video name'][0],
                                                'fatigue level':fatigue_level
                                                }, ignore_index=True)
                    frame_counter = frame_counter + 1
                
                elif video_row['Train/Test'] == 'validation':
                    # save the image to the test output directory 
                    cv2.imwrite((saving_path+'/Validation_Data/'+
                                 video_row['Video name']+'_'+
                                 str(frame_counter)+'.jpg'), 
                                frame)
                    # calculate the fatigue level for the current frame
                    fatigue_level = fatigue_level_calculation(video_row['Video name'], 
                                                              frame_counter, 
                                                              fps, 
                                                              frame_count/10)
                    # update the test data frame with the current frame information 
                    validation_df = validation_df.append({'image name':video_row['Video name']+'_'+str(frame_counter),
                                                'label':video_row['Video name'][0],
                                                'fatigue level':fatigue_level
                                                }, ignore_index=True)
                    frame_counter = frame_counter + 1
                
                
                
                else: 
                    # raise error if no  the data categorey is neither Train nor Test
                    raise ValueError('Error: Unknow data categorey')

                # 20 is in milliseconds, try to increase the value, say 50 and observe
                key = cv2.waitKey(20)
                if (key == ord('q')) or (frame_counter > 5*60*fps):
                    frame_counter = 0
                    break
            else:
                break
        
        # Release the video capture object
        vid_capture.release()
        cv2.destroyAllWindows()
    train_df.to_csv(saving_path+'/'+'train_labels.csv')
    test_df.to_csv(saving_path+'/'+'test_labels.csv')
    validation_df.to_csv(saving_path+'/'+'validation_labels.csv')
    return     


        
video_list_path = '/home/youssef/Phd_Work/Fatigue_Classification/Data/video_list.csv'        
video_main_path = '/home/youssef/Phd_Work/Fatigue_Classification/Data/vidoes'
saving_path = '/home/youssef/Phd_Work/Fatigue_Classification/baseline_classifier/Data'

resampling_labeling(video_list_path, video_main_path, saving_path)       
        