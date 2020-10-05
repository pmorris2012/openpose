import os
import cv2

def check_video(file):
    video = cv2.VideoCapture(file)
    frames_remaining, frame = video.read()
    return frames_remaining

def check_image(file):
    return cv2.haveImageReader(file)

def find_images_videos(directory, files):
    images, videos = [], []
    for file in files:
        file_path = os.path.join(directory, file)
        if check_image(file_path):
            images.append(file_path)
        elif check_video(file_path):
            videos.append(file_path)

    return images, videos

def get_video_properties(video):
    return {
        'fps': video.get(cv2.CAP_PROP_FPS),
        'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frames': int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    }
