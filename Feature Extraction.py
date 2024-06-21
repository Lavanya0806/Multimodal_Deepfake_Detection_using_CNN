import moviepy.editor as mp
import os

# Function to extract audio from video
def extract_audio(video_path, output_path, start_time, end_time):
    video = mp.VideoFileClip(video_path)
    audio = video.audio.subclip(start_time, end_time)
    audio.write_audiofile(output_path)
    video.close()

# Function to process each folder
def process_folder(folder_path):
    # Create a directory to store audio files
    audio_folder = os.path.join('audio', os.path.basename(folder_path))
    os.makedirs(audio_folder, exist_ok=True)

    # List all video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        output_path = os.path.join(audio_folder, os.path.splitext(video_file)[0] + '.wav')

        # Extract audio from video (start time = 0s, end time = 3s)
        extract_audio(video_path, output_path, 0, 3)

# Process each folder in the 'data' directory
data_folder = 'data'
folder_list = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
for folder in folder_list:
    process_folder(folder)




import cv2 
import numpy as np 

import os

folder_list = os.listdir('data')

for folder in folder_list:
        
        # create a path to the folder
        path = 'data/' + str(folder)
        img_files = os.listdir(path)
        print(path)
        cnt=0
        for file in img_files:
	
            #imgpath = ds_path +'\\'+ file
            src = os.path.join(path, file)
            
            # Create a VideoCapture object and read from input file
            cap = cv2.VideoCapture(src)
            # Read until video is completed
            cn=0
            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if cn == 10:
                    # Display the resulting frame
##                    cv2.imshow('Frame', frame)
                    outpath='./img/'+str(folder)+'/'+file+str(cnt)+'.jpg'
                    cv2.imwrite(outpath,frame)
                    cnt+=1
                    break
                cn+=1
                    # Press Q on keyboard to exit

            # When everything done, release
            # the video capture object
            cap.release()
            # Closes all the frames
            
























##import cv2
##import os
##import moviepy.editor as mp
##
### Function to extract audio from video
##def extract_audio(video_path, output_path, start_time, end_time):
##    video = mp.VideoFileClip(video_path)
##    audio = video.audio.subclip(start_time, end_time)
##    audio.write_audiofile(output_path)
##    video.close()
##
### Process each folder
##def process_folder(folder_path):
##    # Create a directory to store images
##    image_folder = os.path.join('img', os.path.basename(folder_path))
##    os.makedirs(image_folder, exist_ok=True)
##
##    # Create a directory to store audio files
##    audio_folder = os.path.join('audio', os.path.basename(folder_path))
##    os.makedirs(audio_folder, exist_ok=True)
##
##    # List all video files in the folder
##    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
##
##    for video_file in video_files:
##        video_path = os.path.join(folder_path, video_file)
##
##        # Extract audio from video (start time = 0s, end time = 3s)
##        output_audio_path = os.path.join(audio_folder, os.path.splitext(video_file)[0] + '.wav')
##        extract_audio(video_path, output_audio_path, 0, 3)
##
##        # Extract frames from video
##        cap = cv2.VideoCapture(video_path)
##        cnt = 0
##        while cap.isOpened():
##            ret, frame = cap.read()
##            if ret:
####                cv2.imshow('Frame', frame)
##                output_image_path = os.path.join(image_folder, os.path.splitext(video_file)[0] + '_' + str(cnt) + '.jpg')
##                cv2.imwrite(output_image_path, frame)
##                cnt += 1
##            else:
##                break
##            if cv2.waitKey(25) & 0xFF == ord('q'):
##                break
##
##        # Release video capture
##        cap.release()
##        cv2.destroyAllWindows()
##
### Process each folder in the 'data' directory
##data_folder = 'data'
##folder_list = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
##for folder in folder_list:
##    process_folder(folder)
