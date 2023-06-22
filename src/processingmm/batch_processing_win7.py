import os
import re
import shutil
import traceback

from processingmm.helpers import load_filenames_raw_data
from processingmm.batch_processing import find_all_folders, get_df_processing, get_to_process

def move_to_NAS(directories: list, processed: list, folder_NAS: str):
    
    already_done = set(os.listdir(folder_NAS))
    df = get_df_processing(directories)
    processed = get_to_process(df, run_all = False, inverse = True)
    
    for folder in processed:
        fname = folder.split('\\')[-1]
        if fname in already_done:
            pass
        else:
            shutil.copytree(folder, os.path.join(folder_NAS, fname))

def find_folder_name(root, data_folder, folder_names):
    """
    check if a folder has the correct format: 202x-xx-xx. if yes, add it to the list of folders

    Parameters
    ----------
    root : str
        complete path to the folder
    data_folder : list
        the list with all the folders containing data
    folder_names : list
        the list with the name of all the measurements
    """
    try:
        # check if the folder name format is 202x-xx-xx
        assert len(re.findall("[\d]{4}-[\d]{2}-[\d]{2}", root)) == 1
        x = re.search("[\d]{4}-[\d]{2}-[\d]{2}", root).group(0)
        splitted = root.split(x)
                    
        # if yes, append it to the lists containing the folder names
        data_folder.append(splitted[0] + x + splitted[1].split('\\')[0])
        folder_names.append(x + splitted[1].split('\\')[0])
                    
    except Exception as e:
        pass


def backup_to_B_drive(directories, target):
    
    folders_backup = get_processed_acquisitionned(directories)[0]
    
    data_raw_backup_path = target
    _, folder_names_backup = find_all_folders(data_raw_backup_path, win7 = True)

    to_copy = []
    for folder in folders_backup:
        if folder in set(folder_names_backup):
            pass
        else:
            to_copy.append(os.path.join(directories[0], folder))
        
    for folder in to_copy:
        source_dir = folder
        destination_dir = os.path.join(data_raw_backup_path[0], source_dir.split('\\')[-1])
        try:
            shutil.copytree(source_dir, destination_dir)
        except FileExistsError:
            pass
        
def get_processed_acquisitionned(directories):
    data_folder, folder_names = find_all_folders(directories, win7 = True)
    filenames_raw = load_filenames_raw_data()
    fn = list(set(filenames_raw))
    fn.sort()
 
    folder_names_acquisitionned = []
    data_folder_acquisitionned = []
    folder_names_not_acquisitionned = []
    folder_names_processed = []
    data_folder_processed = []
    
    for (folder, fname) in zip(data_folder, folder_names):
        
        
        fold = list(set(os.listdir(folder)))
        fold.sort()
            
        if fold == fn:
            with open(os.path.join(folder, 'Logbook.txt')) as f:
                lines = f.readlines()
            if 'Acquisition done and Mueller matrix calculated' in lines[-1]:
                folder_names_acquisitionned.append(fname)
                data_folder_acquisitionned.append(folder)
            else:
                folder_names_not_acquisitionned.append(fname)
        else:
            folder_names_processed.append(fname)   
            data_folder_processed.append(folder)
    
    return folder_names_acquisitionned, data_folder_acquisitionned, folder_names_not_acquisitionned, folder_names_processed, data_folder_processed


def move_back_the_folders(base_directory, base_temp_directory):
    processed = []
    for folder in os.listdir(base_temp_directory):
        folder_trgt = os.path.join(base_directory[0], folder)
        try:
            shutil.rmtree(folder_trgt)
            shutil.move(os.path.join(base_temp_directory, folder), folder_trgt)
            processed.append(folder_trgt)
        except FileNotFoundError:
            pass
        except:
            traceback.print_exc()
    return processed
    