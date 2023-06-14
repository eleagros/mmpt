import os
import shutil
from processingmm.helpers import load_filenames_results, load_filenames_50x50, load_filenames

def move_raw_data_folders(directory: list, measurements_directory: str, Flag = False):
    """
    moves the raw data folders into the newly created raw_data directory

    Parameters
    ----------
    directory : list of str
        the folder name
    measurements_directory : str
        the path to the measurement directory
    """
    path_directory = os.path.join(measurements_directory, directory)
    folders = os.listdir(path_directory)
    
    # check if raw_data folder exists, if not, creates it 
    raw_data_dir = os.path.join(path_directory, 'raw_data')

    if 'raw_data' in folders:
        pass
    else:
        os.mkdir(raw_data_dir)
        
    # get a list of folders containing raw data
    raw_data_folders = get_raw_data_folder(os.listdir(path_directory))

    # move the raw_data_folders to the raw_data directory of the folder
    for raw_data_f in raw_data_folders:
        try:
            shutil.move(os.path.join(path_directory, raw_data_f), raw_data_dir)
        except Exception as e:
            if 'Destination path' in str(e) and 'already exists' in str(e):
                pass
            else:
                print(e)
        
    # also move 'Logbook.txt' & 'TemperatureBook.txt' to the raw_data directory
    try:
        if 'Logbook.txt' in folders:
            shutil.move(os.path.join(path_directory, 'Logbook.txt'), raw_data_dir)
    except Exception as e:
            if 'Destination path' in str(e) and 'already exists' in str(e):
                pass
            else:
                print(e)
     
    try:
        if 'TemperatureBook.txt' in folders:
            shutil.move(os.path.join(path_directory, 'TemperatureBook.txt'), raw_data_dir)

    except Exception as e:
            if 'Destination path' in str(e) and 'already exists' in str(e):
                pass
            else:
                print(e)

    if Flag:
        print('Raw data folders moved')


def get_raw_data_folder(folders: list):
    """
    get all the folders containing raw data (i.e. folder of the following form: xxxnm)

    Parameters
    ----------
    folders : list of str
        the list of folders to be considered
    
    Returns
    -------
    raw_data : list of str
        the list of folders containing raw data
    """
    raw_data = []
    for f in folders:
        if len(f) < 4:
            pass
        else:
            if f[0:2].isdecimal() and f[3:5] == 'nm':
                raw_data.append(f)
            else:
                pass
    return raw_data

def create_directories(measurements_directory: str, directory: list, directories_tbc: list, wavelenghts: list, Flag = False):
    """
    create the architecture for the folders copied from the IMP (polarimetry, 50x50_images, histology,...)

    Parameters
    ----------
    measurements_directory : str
        the path to the measurement directory
    directory : list of str
        the folder name
    directories_tbc : list of str
        the list of directories to be created
    wavelenghts : list of str
        the wavelengths of interest
    Flag : boolean
        indicates wether the progression should be displayed
    """
    path_directory = os.path.join(measurements_directory, directory)

    for directory_tbc in directories_tbc:
        # if the directory already exists...
        if directory_tbc in os.listdir(path_directory):
            
            # ... and if it is polarimetry
            if directory_tbc == 'polarimetry':
                dir_tb = os.path.join(path_directory, directory_tbc)
                for dir_ in os.listdir(dir_tb):
                    
                    # check if 50x50_images exist for all wavelengths or create it
                    if '50x50_images' in os.listdir(os.path.join(dir_tb, dir_)):
                        pass
                    else:
                        os.mkdir(os.path.join(dir_tb, dir_, '50x50_images'))
        
        else:
            # if not, create the directory
            os.mkdir(os.path.join(path_directory, directory_tbc))
            
        if directory_tbc == 'photo' or directory_tbc == 'histology' or directory_tbc == 'annotation':
            pass
        else:
            
            # check if all the wavelgnth sub-directories have been created
            if len(os.listdir(os.path.join(path_directory, directory_tbc))) == len(wavelenghts):
                pass
            else:
                
                # else, create the missing sub-directories
                if len(os.listdir(os.path.join(path_directory, directory_tbc))) == 0:
                    for wavelenght in wavelenghts:
                        os.mkdir((os.path.join(path_directory, directory_tbc, wavelenght)))
                else:
                    for wavelenght in wavelenghts:
                        if wavelenght in os.listdir(os.path.join(path_directory, directory_tbc)):
                            pass
                        else:
                            os.mkdir(os.path.join(path_directory, directory_tbc, wavelenght))
    
    if Flag:
        print('Directories created')


def remove_old_computation(measurements_directory: str, directory: str, Flag = False):
    """
    create the architecture for the folders copied from the IMP (polarimetry, 50x50_images, histology,...)

    Parameters
    ----------
    measurements_directory : str
        the path to the measurement directory
    directory : str
        the folder name
    Flag : boolean
        indicates wether the progression should be displayed
    """
    polarimetry = os.path.join(measurements_directory, directory, 'polarimetry')
    for wl in os.listdir(polarimetry):
        folder = os.path.join(polarimetry, wl)
        filenames = load_filenames()
        filenames_50x50 = load_filenames_50x50()
        filenames_results = load_filenames_results()
            
        for file in os.listdir(folder):
            if file in filenames:
                pass
            else:
                if file == '50x50_images' or file == 'results':
                    if file == '50x50_images':
                        file_res = filenames_50x50
                    else:
                        file_res = filenames_results
                        
                    for file_ind in os.listdir(os.path.join(folder, file)):
                        if file_ind in file_res:
                            pass
                        else:
                            os.remove(os.path.join(folder, file, file_ind))
                else:
                    if file.endswith('_realsize.png'):
                        pass
                    else:
                        if file in filenames:
                            pass
                        else:
                            os.remove(os.path.join(folder, file))
    
    if Flag:
        print('Old computations were removed')


def move_50x50_images(measurements_directory: str, directory: str, Flag = False):
    """
    move the 50x50 images that were already computed in thw new 50x50_images folder

    Parameters
    ----------
    measurements_directory : str
        the path to the measurement directory
    directory : str
        the folder name
    """
    source = os.path.join(measurements_directory, directory, '50x50_images')
    
    try:
        for wl in os.listdir(source):
            target = os.path.join(measurements_directory, directory, 
                                                  'polarimetry', str(wl), '50x50_images')
            for file in os.listdir(os.path.join(source, wl)):
                dest = shutil.move(os.path.join(source, wl, file), os.path.join(target)) 
    except FileNotFoundError:
        pass
        
    try:
        shutil.rmtree(source)
    except:
        pass
    
    try:
        shutil.rmtree(source.replace('50x50_images', 'photo'))
    except:
        pass

    if Flag:
        print('50x50 images were moved')