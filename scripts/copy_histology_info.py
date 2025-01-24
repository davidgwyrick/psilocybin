import shutil
import os
import argparse

import os
import shutil

MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB

# Create the argument parser
parser = argparse.ArgumentParser(description='Copy files and folders from one directory to another')

# Add the source directory argument
parser.add_argument('--mouseID', type=str, nargs='?', default='mouse678913', help='mouseID in form mouse#####')

# Parse the arguments
args = parser.parse_args()

def copy_files(src_path, dst_path):
    """
    Copy files and subdirectories from the source path to the destination path.
    """
    files_to_copy = ['metrics.csv','probe_info.json']
    for item in os.listdir(src_path):
        item_path = os.path.join(src_path, item)

        if os.path.isfile(item_path):
            
            if item in files_to_copy:
                print(f'Copying: {item_path}')
                shutil.copy(item_path, dst_path)

        elif os.path.isdir(item_path):
            # create directory in the destination path if it does not exist
            new_dst_path = os.path.join(dst_path, item)
            if not os.path.exists(new_dst_path):
                os.makedirs(new_dst_path)
                print(f'Creating directory {item}')
            # recursively copy files and subdirectories to the new destination path
            copy_files(item_path, new_dst_path)


if __name__ == '__main__':
    
    mID = args.mouseID 
    print(mID)
    # Get the paths to the source and destination directories from the arguments
    # ServerDir = '/allen/programs/braintv/workgroups/tiny-blue-dot/zap-n-zip/EEG_exp/'
    ServerDir = '/allen/programs/mindscope/workgroups/templeton-psychedelics/'
    LocalDir  = '/data/tiny-blue-dot/zap-n-zip/EEG_exp'
    mouseID_folders = os.listdir(ServerDir)

    if mID not in mouseID_folders:
        print(f"Error: Source directory '{mID}' does not exist on Allen")
        exit()

    #This is the base of the folder tree we're going to copy
    src_dir = os.path.join(ServerDir,mID)

    # Get a list of subfolders in the source directory
    subfolders = sorted([f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))])

    # Print the list of subfolders and prompt the user to select one
    print("Select a subfolder to copy:")
    for i, subfolder in enumerate(subfolders):
        print(f"{i+1}. {subfolder}")
    print('Choose subfolder: ',end=' ')
    selected_subfolder_index = int(input()) - 1

    if selected_subfolder_index < 0 or selected_subfolder_index >= len(subfolders):
        print("Error: Invalid subfolder selection")
        exit()
    selected_subfolder = subfolders[selected_subfolder_index]

    # Create the full path to the selected subfolder in the source directory
    src_path = os.path.join(src_dir, selected_subfolder)
    dst_path = os.path.join(LocalDir,mID,selected_subfolder)

    if not os.path.exists(dst_path):
        try:
            os.makedirs(dst_path)
        except Exception as e:
            print(f"Error making destination directory '{dst_path}': {e}")
            exit()

    #Recursively copy files and folders from src_path to dst_path
    copy_files(src_path, dst_path)
    print("Files copied successfully")



    # # Check if source path and subfolder exist
    # if not os.path.exists(src_path):
    #     print(f"Error: Source path '{src_path}' does not exist")
    # elif subfolder_name and not os.path.exists(os.path.join(src_path, subfolder_name)):
    #     print(f"Error: Subfolder '{subfolder_name}' does not exist in the source path")
    # else:
    #     # Copy files and subfolders to the destination path
    #     if subfolder_name:
    #         src_path = os.path.join(src_path, subfolder_name)
    #     copy_files(src_path, dst_path)
    #     print("Files copied successfully")


    # # Iterate over each file and directory in the selected subfolder
    # for file_name in files:
    #     # Create the full path to the file or directory
    #     src_path = os.path.join(src_subfolder, file_name)
    #     dst_path = os.path.join(dst_subfolder, file_name)

    #     #Skip if file already exists on target machine
    #     if os.path.exists(dst_path):
    #         print(f'Skipping {src_path}')
    #         continue

    #     # If the file or directory is a directory, use shutil.copytree() to copy it recursively
    #     if os.path.isdir(src_path):
    #         try:
    #             shutil.copytree(src_path, dst_path)
    #             print(f'Creating directory {file_name}')
    #         except Exception as e:
    #             print(f"Error copying directory '{src_path}': {e}")

    #     # If the file or directory is a file, use shutil.copy2() to copy it
    #     elif os.path.isfile(src_path):
    #         file_size = os.path.getsize(src_path)
    #         if file_size <= MAX_FILE_SIZE:
    #             try:
    #                 print(f'Copying: {src_path}')
    #                 shutil.copy2(src_path, dst_path)
    #             except Exception as e:
    #                 print(f"Error copying file '{src_path}': {e}")
    #         else:
    #             print(f"Skipping file '{src_path}' because it is larger than {max_GB} Gb")
    #             # Create an empty file with
    #             with open(dst_path, 'w') as f:
    #                 pass  # this creates an empty file
