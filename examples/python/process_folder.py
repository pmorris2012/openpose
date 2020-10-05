import sys
import cv2
import os
import pyopenpose as op
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', default="/Videos")
parser.add_argument('--output_folder', default="/Videos_Pose")
parser.add_argument('--face', dest='face', action='store_true')
parser.add_argument('--hand', dest='hand', action='store_true')
args = parser.parse_args()

pose_dir = os.path.join(args.output_folder, "Pose")
black_pose_dir = os.path.join(args.output_folder, "Black_Pose")
coords_dir = os.path.join(args.output_folder, "Coords")

#os.walk recursively goes through all the files in our args.input_folder
input_paths = []
for directory, folders, files in os.walk(args.input_folder):
    for file in files:
        input_paths.append(os.path.join(directory, file))

print("found", len(input_paths), "input files")
print("example:", input_paths[0])

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

def do_openpose(image):
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop([datum])
    return datum

def is_empty(keypoint):
    return keypoint.sum() == 0

def draw_point(image, x, y, color):
    radius = int(min(image.shape[0], image.shape[1]) * .005)
    cv2.circle(image, (x, y), radius, color, thickness=radius, lineType=8, shift=0)

def draw_line(image, x1, y1, x2, y2, color):
    thickness = max(2, int(min(image.shape[0], image.shape[1]) / 200.0))
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

Body25Pairs = [
    (1,8), (1,2), (1,5), (2,3), (3,4), (5,6), (6,7), (8,9), (9,10), (10,11), (8,12), 
    (12,13), (13,14), (1,0), (0,15), (15,17), (0,16), (16,18), (14,19), (19,20), 
    (14,21), (11,22), (22,23), (11,24)
]

HandPairs = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), 
    (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20)
]

FacePairs = [
    (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11),
    (11,12), (12,13), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20), (20,21), 
    (22,23), (23,24), (24,25), (25,26), (27,28), (28,29), (29,30), (31,32), (32,33), 
    (33,34), (34,35), (36,37), (37,38), (38,39), (39,40), (40,41), (41,36), (42,43), 
    (43,44), (44,45), (45,46), (46,47), (47,42), (48,49), (49,50), (50,51), (51,52), 
    (52,53), (53,54), (54,55), (55,56), (56,57), (57,58), (58,59), (59,48), (60,61), 
    (61,62), (62,63), (63,64), (64,65), (65,66), (66,67), (67,60)
]

def get_color(i, mode):
    if mode == "face":
        return [255, 255, 255] #white
    
    return CocoColors[i % len(CocoColors)]

pairs_dict = {
    "body": Body25Pairs,
    "hand": HandPairs,
    "face": FacePairs
}

def array_empty(array):
    return len(array.shape) == 0

def draw_keypoints(image, keypoints, mode="body"):
    pairs = pairs_dict[mode]
    
    if array_empty(keypoints):
        return image
    
    for human in keypoints:
        for i, keypoint in enumerate(human):
            if not is_empty(keypoint):
                point_color = get_color(i, mode)
                draw_point(image, int(keypoint[0].round()), int(keypoint[1].round()), point_color)

        for i, pair in enumerate(pairs):
            keypoint0 = human[pair[0]]
            keypoint1 = human[pair[1]]
            if not is_empty(keypoint0) and not is_empty(keypoint1):
                line_color = get_color(i, mode)
                draw_line(image, int(keypoint0[0].round()), int(keypoint0[1].round()), int(keypoint1[0].round()), int(keypoint1[1].round()), line_color)

    return image

def draw_keypoints_from_result(image, result):
    image_pose = draw_keypoints(image, result.poseKeypoints)
    image_pose = draw_keypoints(image_pose, result.faceKeypoints, mode="face")
    image_pose = draw_keypoints(image_pose, result.handKeypoints[0], mode="hand")
    return draw_keypoints(image_pose, result.handKeypoints[1], mode="hand")

def get_black_image(image):
    image = np.copy(image)
    image[:,:,:] = 0
    return image

# '\r' is a "carriage return"
# it moves the cursor back to the biginning of the line
# the next line will then overwrite the previus one
def log_video_progress(video):
    frame_idx = video.get(cv2.CAP_PROP_POS_FRAMES)
    sys.stdout.write(str(int(frame_idx)) + " frames\r")
    sys.stdout.flush()

def decode_fourcc(codec):
    return ''.join([
        chr(codec & 255),
        chr((codec >> 8) & 255),
        chr((codec >> 16) & 255),
        chr((codec >> 24) & 255)])

def move_path(path, folder_from, folder_to):
    return path.replace(folder_from, folder_to, 1)

def create_dirs(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_coords(path, result):
    pose = np.empty(shape=(0,25,3)) if array_empty(result.poseKeypoints) else result.poseKeypoints
    face = np.empty(shape=(0,70,3)) if array_empty(result.faceKeypoints) else result.faceKeypoints
    handl = np.empty(shape=(0,21,3)) if array_empty(result.handKeypoints[0]) else result.handKeypoints[0]
    handr = np.empty(shape=(0,21,3)) if array_empty(result.handKeypoints[1]) else result.handKeypoints[1]
    
    np.savez(path, pose=pose, face=face, handl=handl, handr=handr)

for input_path in input_paths:
    video = cv2.VideoCapture(input_path)
    
    framerate = video.get(cv2.CAP_PROP_FPS)
    codec = int(video.get(cv2.CAP_PROP_FOURCC))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pose_path = move_path(input_path, args.input_folder, pose_dir)
    create_dirs(pose_path)
    video_pose = cv2.VideoWriter(pose_path, cv2.VideoWriter_fourcc(*'MPEG'), framerate, (width, height))
    
    black_pose_path = move_path(input_path, args.input_folder, black_pose_dir)
    create_dirs(black_pose_path)
    video_black_pose = cv2.VideoWriter(black_pose_path, cv2.VideoWriter_fourcc(*'MPEG'), framerate, (width, height))
    
    coords_path = move_path(input_path, args.input_folder, coords_dir)
    coords_path, _ext = os.path.splitext(coords_path)
    create_dirs(os.path.join(coords_path, "test"))
    
    frames_remaining, frame = video.read()
    frame_idx = 0
    while frames_remaining:
        result = do_openpose(frame)

        image_pose = draw_keypoints_from_result(frame, result)
        video_pose.write(image_pose)

        black = get_black_image(frame)
        black_pose = draw_keypoints_from_result(black, result)
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
