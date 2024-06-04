# %%
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import glob
# %%
class DataFolder:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.metadata = self.load_metadata()
        self.files = self.metadata.get('files', [])

    def load_metadata(self):
        metadata_path = os.path.join(self.folder_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as json_file:
                return json.load(json_file)
        else:
            return self.create_metadata()

    def create_metadata(self):
        # Method to create initial metadata
        metadata = {
            "description": "Description of the data and its purpose.",
            "source": "Where the data was obtained from.",
            "date_collected": "YYYY-MM-DD",
            "files": [],
            "structure": {
                "columns": []
            },
            "metadata": {
                "created_by": "Your name",
                "created_on": datetime.now().strftime('%Y-%m-%d'),
                "last_modified": datetime.now().strftime('%Y-%m-%d')
            }
        }
        
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            if os.path.isfile(file_path) and file_name != 'metadata.json':
                file_info = {
                    "file_name": file_name,
                    "file_type": file_name.split('.')[-1],
                    "size": os.path.getsize(file_path),
                    "notes": ""
                }
                metadata['files'].append(file_info)
        
        with open(os.path.join(self.folder_path, 'metadata.json'), 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        
        return metadata

    def list_files(self):
        return [file['file_name'] for file in self.files]

    def plot_data(self, file_name):
        file_path = os.path.join(self.folder_path, file_name)
        if not os.path.exists(file_path):
            print(f"File {file_name} does not exist.")
            return

        # Example: Assuming the data file is a CSV and contains columns 'x' and 'y'
        if file_name.endswith('.csv'):
            data = pd.read_csv(file_path)
            if 'x' in data.columns and 'y' in data.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(data['x'], data['y'])
                plt.title(f"Plot for {file_name}")
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()
            else:
                print(f"Columns 'x' and 'y' not found in {file_name}.")
        else:
            print(f"File type of {file_name} is not supported for plotting.")
    
    def update_metadata(self):
        self.metadata['metadata']['last_modified'] = datetime.now().strftime('%Y-%m-%d')
        with open(os.path.join(self.folder_path, 'metadata.json'), 'w') as json_file:
            json.dump(self.metadata, json_file, indent=4)

# Example usage
folder_path = '/path/to/your/folder'
data_folder = DataFolder(folder_path)

# List files
print(data_folder.list_files())

# Plot data from a specific file
data_folder.plot_data('data_file.csv')

# %%
pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5')

# %%
pth = Path('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5')
# check levels of folders
def count_subdir_levels(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"The provided path {path} is not a directory")
    
    def count_levels(directory, current_level):
        max_level = current_level
        for subdir in directory.iterdir():
            if subdir.is_dir():
                max_level = max(max_level, count_levels(subdir, current_level + 1))
        return max_level

    return count_levels(path, 0)

level_count = count_subdir_levels(pth)

if level_count > 2:
    print(f"Error: The directory {pth} has more than 2 levels of subdirectories.")
    print(f'Cannot datify {pth}')    
    
if level_count == 1:
    datafied = {}
    
if level_count == 2:
    datafied_list = []

    for subpth in pth.iterdir():
        # always ignore . files
        datafied = {}
        if str(subpth.stem).startswith('.'):
            continue       
        
        print(f'- {subpth.stem}')
        
        for subsubpth in subpth.iterdir():
            if str(subsubpth.stem).startswith('.'):
                continue
            print(f'   - {subsubpth.stem}')
        

    
        



# %%

deepest_dir = Path('/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5/NonIntersectingBox-N500-AR100-Scale1_20240531-222436/rodsContactPlots_20240602-232751')

extensions = set()
filename_length = set()
for files in deepest_dir.iterdir():

    
    # TEST 1: check if there is a dir
    if files.is_dir():
        print(f'Found a dir: {files.stem}. Prune!')
        break
    
    # add extension
    extensions.add(files.suffix)
    # TEST 2: check if new extension was introduced
    if len(extensions) > 1:
        print(f'Found multiple extensions: {extensions}')
        break
    
    # TEST 3: File name length. Should be strict about it.
    filename_length.add(len(files.stem))    
    if len(filename_length) > 1:
        print(f'Found multiple filename lengths: {filename_length}')
        break
    
    # 
    
    # str(files.stem).split('_')

    
    
    # data_file_paths = list(deepest_dir.glob('**/*'))
    
    
    
    
    
    
    
    
        
    
        
    

