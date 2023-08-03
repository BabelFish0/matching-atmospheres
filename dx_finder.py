import numpy as np

def get_bin_widths(values, val_min=None, val_max=None):
    '''
    Find the difference between the midpoints of each consecutive value in `values`. If no upper or lower bound is given, the midpoint between element 0 and 1 will be assumed symmetrical. This is also true for element -1 and -2.
    
    -----
    ## Inputs
    - `values`: 1D numpy ndarray of values of which the 'width' is to be determined.
    - `val_min`: lower limit - if the symmetric midpoint for the first or last element is less than this, `val_min` will be used instead to find the width.
    - `val_max`: upper limit - if the symmetric midpoint for the first or last element is greater than this, `val_max` will be used instead to find the width.
    -----
    ## Returns
    - `dx_values`: 1D numpy ndarray of widths for every element in `values`.
    '''
    midpoints = .5 * (values[1:] + values[:-1])
    midpoints = np.insert(midpoints, 0, 2*values[0]-midpoints[0])
    midpoints = np.append(midpoints, 2*values[-1]-midpoints[-1])
    if val_min != None or val_max != None:
        midpoints = np.clip(midpoints, val_min, val_max)
    dx_values = midpoints[1:] - midpoints[:-1]
    return dx_values

print(get_bin_widths(np.array([0, 0.06, 0.2, 1.0]), 0, 1)) # cloud, consistent
print(get_bin_widths(np.array(['-2.3', '-1.0', '+0.0', '+1.0', '+1.7', '+2.0', '+2.3']).astype(float))) # metal, not consistent with the original dx values