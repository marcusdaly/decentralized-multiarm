from pathlib import Path
from os.path import abspath
from json import load
import os

root_dir = 'tasks/'

files = [abspath(file_name)
                          for file_name in Path(root_dir).rglob('*.json')
                          if 'config' not in str(file_name)]

f = open("custom_benchmark/all.txt","w+")

task_ct = 0
for file in files:

    task_file = load(open(file))
    start_config = task_file['start_config']
    ur5_count = len(start_config)
    if ur5_count == 2:
        file_name = str(os.path.basename(file))
        f.write("%s\r" % file_name)

        os.system(f'cp {file} custom_benchmark/{file_name}')
        task_ct += 1
        if task_ct >= 100:
            break

f.close()
