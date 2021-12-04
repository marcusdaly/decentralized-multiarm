from pathlib import Path
from os.path import abspath
from json import load
import os

root_dir = '/home/leo/Documents/robot_learning/decentralized-multiarm/tasks/'

files = [abspath(file_name)
                          for file_name in Path(root_dir).rglob('*.json')
                          if 'config' not in str(file_name)]

f = open("all.txt","w+")

for file in files:
    task_file = load(open(file))
    start_config = task_file['start_config']
    ur5_count = len(start_config)
    if ur5_count == 2:
        file_name = str(os.path.basename(file))
        f.write("%s\r" % file_name)

f.close()
