import os

# Specify your folder path
folder_path = 'C:/Users/charoensupthawornt/Work/Securade.ai/hub/labels/train/new'

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    # Skip directories, process files only
    if not os.path.isfile(os.path.join(folder_path, filename)):
        continue

    # Find the first hyphen
    if '-' in filename:
        # Keep everything after the first hyphen
        new_name = filename.split('-', 1)[1]
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {filename}  -->  {new_name}')
    else:
        print(f'No hyphen in: {filename} (skipped)')