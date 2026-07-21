import numpy as np
from vispy import color
from sklearn.preprocessing import MinMaxScaler
# import xarray as xr


def fill_value_in_range(array, index_array, value=255.):
    """
    The function fill the value in a given array using indexing array.
    The indexing array can have the index outside of the given array. The position indicated by index array is filled with the value.
    The outside index will be disregarded.
    array: numpy array to be filled with value.
    index_array: numpy array for indexing. Each column has the position of the array to be filled.
    value: value to be filled. int or numpy array with the size of column number in index_array.
    """
    dim = array.ndim
    # Select indices within the range of the array.
    column_to_select = np.all(
        [np.all([index_array[i, :] >= 0 for i in range(dim)], axis=0),
         np.all([index_array[i, :] < (array.shape[i] - 0.5) for i in range(dim)], axis=0)],
        axis=0
    )
    select_idx = index_array[:, column_to_select]
    select_idx = np.around(select_idx).astype(int)

    # Fill the value in the array.
    if isinstance(value, np.ndarray):
        array[tuple([select_idx[i, :] for i in range(dim)])] = value[column_to_select]
    else:
        array[tuple([select_idx[i, :] for i in range(dim)])] = value

    return array


def create_image_from_position_and_values(positions, values, size=None, default_value=0):
    """
    Create an image array that is filled with values on positions. 
    Arguments:
        positions (ndarray): positions where values will be filled.
        values (ndarray): The row size of the values and the row size of the positions should be the same. The values can be a vector.
        size (tuple): The size of the image. The dimension of the values will be added to the last dimension of the image followed after the dimension indicated in the size. 
    Return:
        img (ndarray): The last dimension is the same as columns of the values.  
    """
    if type(values) == (int or float):
        if size is None:
            img = np.full(tuple(positions.astype(int).max(axis=0)+1), default_value, dtype=float)
        else:
            img = np.full(size, default_value, dtype=float)
        
        img[tuple(positions.T.astype(int))] = values
        
    else:
        if values.ndim == 1:
            values = values[:,np.newaxis]

        if size is None:
            img = np.full(tuple(positions.astype(int).max(axis=0)+1)+(values.shape[-1],), default_value, dtype=float)
        else:
            img = np.full(size + (values.shape[-1],), default_value, dtype=float)

        img[tuple(positions.T.astype(int))] = values
    
    return img


def visualize_in_original_space(arr_pos, arr_val, shape=None):
    """
    Arguments
        arr_pos (xarray): the positional array. 2/3xNxM. space axis can be specified
        arr_val (xarray): the value array with shared coordinates with position
    Return:
        ndarray: the image filled with the value of arr_val in the position indicated in arr_pos
    """
    positions = arr_pos.copy()
    positions = positions.fillna(-1)
    positions = positions.stack(pos=['time', 'track'])
    shape_axis = positions.dims.index('space')

    values = arr_val.copy()
    values = values.stack(pos=['time', 'track'])

    if shape is not None:
        img = np.zeros(shape, dtype=arr_val.dtype)
    else:
        shape = np.ceil(positions.data.max(1 - shape_axis)).astype(int)
        img = np.zeros(shape, dtype=arr_val.dtype)

    if shape_axis == 1:
        positions = positions.T

    # clipping
    positions = positions.data
    positions[positions <= 0] = 0
    for i, m in enumerate(shape):
        positions[i, positions[i, :] >= m - 1] = m - 1
    positions = positions.astype(int)

    img[tuple(positions)] = values.data

    return img


def min_max_scaling(arr1d):
    """
    """
    return MinMaxScaler().fit(arr1d[:,np.newaxis]).transform(arr1d[:,np.newaxis]).squeeze()


def vector_color_mapping(arr1d, colormap='plasma',scaling=True, low_p=0, high_p=100):
    """
    """
    if scaling:
        low = np.percentile(arr1d, low_p)
        high = np.percentile(arr1d, high_p)
        scaled_arr1d = (arr1d-low)/(high-low)
    else:
        scaled_arr1d = arr1d
    return color.get_colormap(colormap).map(scaled_arr1d)

