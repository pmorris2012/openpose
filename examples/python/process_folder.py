import sys
import cv2
import os
import pyopenpose as op
import numpy as np
import argparse

from drawing import draw_keypoints

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', default="/Videos")
parser.add_argument('--output_folder', default="/Videos_Pose")
parser.add_argument('--face', dest='face', action='store_true')
parser.add_argument('--hand', dest='hand', action='store_true')
parser.add_argument('--draw_pose', dest='draw_pose', action='store_true')
parser.add_argument('--draw_black_pose', dest='draw_black_pose', action='store_true')
parser.add_argument('--image_ext', default=".png")
parser.add_argument('--video_ext', default=".mp4")
parser.add_argument('--fourcc_code', default="H264")
args = parser.parse_args()

pose_dir = os.path.join(args.output_folder, "Pose")
black_pose_dir = os.path.join(args.output_folder, "Black_Pose")
coords_dir = os.path.join(args.output_folder, "Coords")

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/openpose/models/"
params["face"] = args.face
params["hand"] = args.hand
params["render_pose"] = 0 #we will manually draw pose, so we turn this off

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def process_image(image):
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop([datum])
    return datum

modes = ['body']
if args.face: 
    modes.append('face')
if args.hand: 
    modes.extend(['handl', 'handr'])

array_dict = {
    "body": lambda result: result.poseKeypoints,
    "face": lambda result: result.faceKeypoints,
    "handl": lambda result: result.handKeypoints[0],
    "handr": lambda result: result.handKeypoints[1]
}

def draw_pose(image, result, modes):
    for mode in modes:
        array = array_dict[mode](result)
        image = draw_keypoints(image, mode=mode)
    return image

# '\r' is a "carriage return"
# it moves the cursor back to the biginning of the line
# the next line will then overwrite the previus one
def log_video_progress(video):
    frame_idx = video.get(cv2.CAP_PROP_POS_FRAMES)
    sys.stdout.write(str(int(frame_idx)) + " frames\r")
    sys.stdout.flush()

def move_path(path, folder_from, folder_to):
    return path.replace(folder_from, folder_to, 1)

def create_dirs(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_coords(path, result, modes):
    arrays = {}
    for mode in modes:
        array = array_dict[mode](result)
        if len(array.shape) > 0:
            arrays[mode] = array
    
    np.savez(path, **arrays)

def check_video(file):
    video = cv2.VideoCapture(file)
    frames_remaining, frame = video.read()
    return frames_remaining

def check_image(file):
    return cv2.haveImageReader(file)

#os.walk recursively goes through all the files in our args.input_folder
for directory, folders, files in os.walk(args.input_folder):
    images, videos = [], []
    for file in files:
        file_path = os.path.join(directory, file)
        if check_image(file_path):
            images.append(file_path)
        elif check_video(file_path):
            videos.append(file_path)

for input_path in input_paths:
    video = cv2.VideoCapture(input_path)
    
    framerate = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pose_path = move_path(input_path, args.input_folder, pose_dir)
    create_dirs(pose_path)
    video_pose = cv2.VideoWriter(pose_path, cv2.VideoWriter_fourcc(*args.fourcc_code), framerate, (width, height))
    
    black_pose_path = move_path(input_path, args.input_folder, black_pose_dir)
    create_dirs(black_pose_path)
    video_black_pose = cv2.VideoWriter(black_pose_path, cv2.VideoWriter_fourcc(*args.fourcc_code), framerate, (width, height))
    
    coords_path = move_path(input_path, args.input_folder, coords_dir)
    coords_path, _ext = os.path.splitext(coords_path)
    create_dirs(os.path.join(coords_path, "test"))
    
    frames_remaining, frame = video.read()
    frame_idx = 0
    while frames_remaining:
        result = process_image(frame)

        image_pose = draw_pose(frame, result)
        video_pose.write(image_pose)

        black = np.zeros_like(frame) #get black image
        black_pose = draw_pose(black, result)
        video_black_pose.write(black_pose)

        coords_frame_path = os.path.join(coords_path, str(frame_idx) + ".npz")
        save_coords(coords_frame_path, result)

        log_video_progress(video)
        frames_remaining, frame = video.read()
        frame_idx += 1
        
    print(input_path)
    video.release()
    video_pose.release()
    video_black_pose.release()
