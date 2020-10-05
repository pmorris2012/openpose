import sys
import cv2
import os
import pyopenpose as op
import numpy as np
import argparse
from tqdm import tqdm

from drawing import draw_keypoints
from file_utils import create_write_dir
from cv_tuils import find_images_videos, get_video_properties

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

FOURCC_CODE = cv2.VideoWriter_fourcc(*args.fourcc_code)

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

def get_keypoints(opWrapper, image):
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop([datum])
    return datum

def draw_pose(image, result, modes):
    for mode in modes:
        array = array_dict[mode](result)
        image = draw_keypoints(image, mode=mode)
    return image

def rescale_coord_array(array, frame_size):
    rescale_dim = np.argmin(frame_size)
    pixel_offset = (max(frame_size) - min(frame_size)) / 2
    array[:,:,rescale_dim] += pixel_offset
    array[:,:,:2] /= max(frame_size)
    return array

def rescale_coords(result, modes, frame_size):
    arrays = {}
    for mode in modes:
        array = array_dict[mode](result)
        if len(array.shape) > 0:
            arrays[mode] = rescale_coord_array(array, frame_size)
    return arrays

def process_image(image_path):
    pass

def process_video(video_path):
    video = cv2.VideoCapture(video_path)
    props = get_video_properties(video)
    frame_size = (props['width'], props['height'])

    if args.draw_pose:
        pose_path = create_write_dir(video_path, pose_dir, ext=args.video_ext)
        video_pose = cv2.VideoWriter(pose_path, FOURCC_CODE, props['fps'], frame_size)
    if args.draw_black_pose:
        black_pose_path = create_write_dir(video_path, black_pose_dir, ext=args.video_ext)
        video_black_pose = video_pose = cv2.VideoWriter(black_pose_path, FOURCC_CODE, props['fps'], frame_size)
        
    coords_path = create_write_dir(video_path, coords_dir, ext='/')
    
    progress_bar = tqdm(initial=1, total=props['frames'], desc=os.path.basename(video_path)[:15])
    frames_remaining, frame = video.read()
    frame_idx = 0
    while frames_remaining:
        result = get_keypoints(frame)

        if args.draw_pose:
            image_pose = draw_pose(frame, result, modes)
            video_pose.write(image_pose)
        if args.draw_black_pose:
            black = np.zeros_like(frame) #get black image
            black_pose = draw_pose(black, result, modes)
            video_black_pose.write(black_pose)

        coords_frame_path = os.path.join(coords_path, str(frame_idx) + ".npz")
        coord_arrays = rescale_coords(result, modes)
        np.savez(coords_frame_path, **coord_arrays)

        frames_remaining, frame = video.read()
        frame_idx += 1
        progress_bar.update(1)
        
    progress_bar.close()
    video.release()
    if args.draw_pose:
        video_pose.release()
    if args.draw_black_pose:
        video_black_pose.release()

#os.walk recursively goes through all the files in args.input_folder
for directory, folders, files in os.walk(args.input_folder):
    print(F"searching {directory}")
    image_paths, video_paths = find_images_videos(directory, files)
    print(F"found {len(image_paths)} images and {len(video_paths)} videos")
    
    if len(image_paths) > 0:
        for image_path in tqdm(image_paths, desc='images'):
            process_image(image_path)

    if len(video_paths) > 0:
        for video_path in tqdm(video_paths, desc='videos'):
            process_video(video_path)
