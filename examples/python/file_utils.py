import os

def move_path(path, folder_from, folder_to):
    return path.replace(folder_from, folder_to, 1)

def replace_ext(file, ext):
    return os.path.splitext(file)[0] + ext

def create_dirs(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_write_dir(path, from_dir, to_dir, ext):
    write_path = move_path(path, from_dir, to_dir)
    write_path = replace_ext(write_path, ext)
    create_dirs(write_path)
    return write_path
