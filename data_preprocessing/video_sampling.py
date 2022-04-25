# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
""" 
import cv2
import os 

def video_sampling(file_path, saving_path, file_list, seconds):
    """ This function takes video as an input and return the frame in seconds.
  

    Parameters
    ----------
    file_path : string
        The path of the input video.
    saving_path : string
        The path to the store the images.
    file_list : string 
        The list of the videos to be resampled.
    seconds : int
        The duration in seconds that will be resampled from the video.

    Returns
    -------
    None.

    """
    videos_list = open(file_list, "r")
    for video in videos_list:
        i = 0
        # Create a video capture object, in this case we are reading the video from a file
        vid_capture = cv2.VideoCapture(os.path.join(file_path,video[0:3]+'.mp4'))
        
        if (vid_capture.isOpened() == False):
            print("Error opening the video file")
        else:
            # Read fps and frame count
            # Get frame rate information
            # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
            fps = vid_capture.get(5)
            print('Frames per second : ', fps,'FPS')
        
            # Get frame count
            # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
            frame_count = vid_capture.get(7)
            print('Frame count : ', frame_count)
        
        while(vid_capture.isOpened()):
            # vid_capture.read() methods returns a tuple, first element is a bool 
            # and the second is frame
            ret, frame = vid_capture.read()
            if ret == True:
                cv2.imshow('Frame',frame)
                # save the image
                cv2.imwrite(os.path.join(saving_path, video[0:3]+'_'+str(i)+'.jpg'), frame)
                i = i + 1
                
                # 20 is in milliseconds, try to increase the value, say 50 and observe
                key = cv2.waitKey(20)
                if (key == ord('q')) or (i > 15*fps):
                    i = 0
                    break
            else:
                break
        
        # Release the video capture object
        vid_capture.release()
        cv2.destroyAllWindows()
        
    

