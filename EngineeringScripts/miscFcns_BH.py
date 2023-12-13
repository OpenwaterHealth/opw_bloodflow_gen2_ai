import os, re

def list_non_folder_dirs(relative_path):
    non_folder_dirs = []
    absolute_path = os.path.abspath(relative_path)
    for dirpath, dirnames, filenames in os.walk(absolute_path):
        if not dirnames:
            if ("FULLSCAN" in dirpath or "LONGSCAN" in dirpath):
                non_folder_dirs.append(os.path.abspath(dirpath))
    return non_folder_dirs

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)