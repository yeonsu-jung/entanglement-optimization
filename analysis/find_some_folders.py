import os
from pathlib import Path

# find subdirs in a directory which doesn't have .mov file in it

parent_folder = Path('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/HangModelos_RodMotion')

subdirs = [x for x in parent_folder.iterdir() if x.is_dir()]
for subdir in subdirs:
    if not any([x.suffix == '.mp4' for x in subdir.iterdir()]):
        # print mov file name
        print(subdir)

