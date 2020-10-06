import os
import cv2

def check_video(file):
    video = cv2.VideoCapture(file)
    frames_remaining, frame = video.read()
    return frames_remaining

def check_image(file):
    return cv2.haveImageReader(file)

def get_video_properties(video):
    if type(video) == str: #if file path, load video
        video = cv2.VideoCapture(video)
        
    return {
        'fps': video.get(cv2.CAP_PROP_FPS),
        'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frames': int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    }
