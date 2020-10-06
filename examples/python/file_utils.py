import os

def move_path(path, folder_from, folder_to):
    return path.replace(folder_from, folder_to, 1)

def replace_ext(file, ext):
    return os.path.splitext(file)[0] + ext

def create_dirs(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
