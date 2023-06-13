import re
import os
import pandas as pd
from processingmm.helpers import load_filenames


def find_all_folders(directories):
    """
    walk through all of the directories present in the folders "directories" given as an input. finds
    the folder with the 202x-xx-xx name format and return the list of folders. separates the transmission
    and reflection measurements in two different lists - processed separatly

    Parameters
    ----------
    directories : list of str
        a list containing the directories to scan

    Returns
    -------
    data_folder : list
        the list with all the folders containing data
    folder_names : list
        the list with the name of all the measurements
    data_folder_transmission : list
        the list with all the folders containing data for transmission
    folder_names_transmission : list
        the list with the name of all the measurements for transmission
    """
    data_folder = []
    folder_names = []

    for directory in directories:
        for root, dirs, files in os.walk(directory, topdown=False):
            if 'TRANSMISSION' in root:
                pass
            else:
                find_folder_name(root, data_folder, folder_names)
    return list(set(data_folder)), list(set(folder_names))

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


def find_processed_folders(data_folder):
    """
    iterates over each of the folder to find the ones containing 550nm/650nm raw data and determine the ones that have been processed

    Parameters
    ----------
    data_folder : list
        the list with all the folders containing data
    transmission : 
        boolean indicating whether or not we are processing transmission data
    
    Returns
    -------
    processed_ : list
        a list of booleans indicating wether the folder has been processed
    data_folder_nm : dict
        a dictionnary indicating wether the data for 550nm, 650nm or both was obtained for each folder
    """
    wavelenghts = ['450nm', '500nm', '550nm', '600nm', '650nm', '700nm']

    processed_nm = {}
    data_folder_nm = {}

    # iterate over each folder containing data
    for path in data_folder:
        data = []
        processed = []
        for wl in wavelenghts:
            # check if raw data is available for the different measurements
            data.append(is_there_data(os.path.join(path, 'raw_data', wl)))
            processed.append(is_processed(path, wl))
        
        # add the information to lists
        processed_nm[path] = processed
        data_folder_nm[path] = data
        
    return processed_nm, data_folder_nm, wavelenghts

def is_there_data(path):
    """
    check if raw data is available for the path given as an input

    Parameters
    ----------
    path : str
        the path to the folder containing the raw data
    
    Returns
    -------
    data_exist : bool
        boolean indicating the presence of one or more .cod files
    """
    data_exist = False
    try:
        data_exist = len(os.listdir(path)) == 2
    except FileNotFoundError:
        data_exist = False
    return data_exist


def is_processed(path, wl):
    """
    check if the folders for which raw data is available have been processed (i.e. contains more than 'treshold' files)

    Parameters
    ----------
    path : str
        the path to the folder to check
    nm550_data : bool
        boolean indicating whether or not data has been obtained for 550nm
    nm650_data : bool
        boolean indicating whether or not data has been obtained for 650nm
    processed : list
        a list of booleans indicating wether the folder has been processed
    prefix : str
        the folder in which the polarimetry data is located ('/polarimetry' by default)
    treshold : int
        the number of files to be found so that the folder is considered as processed (20 by default)
    """
    filenames = load_filenames()
    all_file_names = os.listdir(os.path.join(path, 'polarimetry', wl))

    all_found = True

    for filename in filenames:

        found_file = False

        if filename == '_realsize.png':
            for file in all_file_names:
                if file.endswith(filename):
                    found_file = True
        else:
            for file in all_file_names:
                if file == filename:
                    found_file = True
        
        if not found_file:
            all_found = False

    return all_found


def create_folders_df(data_folder, processed, data_folder_nm, wavelenghts):
    """
    create a dataframe referencing the path to the folder and if it has been processed

    Parameters
    ----------
    data_folder : list
        the list with all the folders containing data
    processed_ : list 
        a list of booleans indicating wether the folder has been processed
    save_files : bool
        a boolean indicating wether or not to save the df as excel file 
    transmission : bool
        boolean indicating whether or not we are processing transmission data
        
    Returns
    -------
    df : pd.dataframe
        a dataframe referencing the path to the folder and if it has been processed
    """
    df_list = []
    for folder in data_folder:
        for idx, boolean in enumerate(processed[folder]):
            df_list.append([folder, boolean, data_folder_nm[folder][idx], wavelenghts[idx]])
            
    df = pd.DataFrame(df_list, columns = ['folder name', 'processed', 'data presence', 'wavelength'])
    df = df[df['data presence']]
    df = df.reset_index(drop=True)
        
    return df