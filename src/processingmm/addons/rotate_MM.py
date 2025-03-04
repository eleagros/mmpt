from scipy import ndimage
import numpy as np

def apply_angle_correction(MM: dict, angle_correction: int) -> None:
    """Applies angle correction to the Mueller matrix."""
    if angle_correction == 0:
        return MM
    else:
        for parameter in MM:
            if parameter == 'azimuth':
                if angle_correction == 90:
                    MM[parameter] = rotate_maps_90_deg(MM[parameter], azimuth = True)
                else:
                    MM[parameter] = ndimage.rotate(MM[parameter], angle = angle_correction, reshape = False)
                    MM[parameter] = (MM[parameter] - angle_correction) % 180
            else:
                MM[parameter] = rotate_parameter(parameter, angle_correction, MM_new = MM)
        return MM
       

def rotate_maps_90_deg(map_resize: np.ndarray, azimuth = False):
    """
    rotate_maps allows to rotate an array by 90 degree

    Parameters
    ----------
    map_resize : array
        the array that will be rotated
    idx_azimuth : boolean
        indicates if we are working with azimuth data (hence correction would be needed, default: False)
    
    Returns
    -------
    resized_rotated : array
        the rotated array
    """
    rotated = np.rot90(map_resize)[0:map_resize.shape[0], :]
    
    resized_rotated = np.zeros(map_resize.shape)
    for idx, x in enumerate(resized_rotated):
        for idy, y in enumerate(x):
            if idy < (map_resize.shape[1] - rotated.shape[0]) / 2 or idy > map_resize.shape[1] - (map_resize.shape[1] - rotated.shape[0]) / 2:
                pass
            else:
                try:
                    if azimuth:
                        resized_rotated[idx, idy] = ((rotated[idx, int(idy - ((map_resize.shape[1] - rotated.shape[0]) / 2 - 1))] + 90) % 180)
                    else:
                        resized_rotated[idx, idy] = rotated[idx, int(idy - ((map_resize.shape[1] - rotated.shape[0]) / 2 - 1))]
                except:
                    pass

    return resized_rotated    
         
                
def rotate_parameter(parameter, angle_correction, MM_new = None):
    if MM_new is None:
        value = parameter
    else:
        value = MM_new[parameter]
        
    if angle_correction == 90:
        value_rotated = rotate_maps_90_deg(value)
    elif angle_correction == 180:
        value_rotated = value[::-1,::-1]
    else:
        if not MM_new is None:
            if parameter == 'Msk':
                rotated = ndimage.rotate(value.astype(float), angle = angle_correction, reshape = False)
                value_rotated = rotated > 0.5
        else:
            rotated = ndimage.rotate(value, angle = angle_correction, reshape = False)
            value_rotated = rotated
            
    return value_rotated