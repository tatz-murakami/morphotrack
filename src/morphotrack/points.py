import numpy as np
from scipy import spatial
from tqdm import tqdm
import xarray as xr


def isin_thickness(half_thickness, position, flow, neighbors):
    """
    The function returns the neighbors within a certain thickness.
        half_thickness (float): half thickness of the column
        position (ndarray): center of the column
        flow (ndarray): axis of the column
        neighbors (ndarray): Nx3 array
    return: neighbors in thickness, boolean for indexing
    """
    # normalize flow

    # judge isin from dot product
    cond1 = np.dot(neighbors - (position + flow * half_thickness), flow) < 0
    cond2 = np.dot(neighbors - (position - flow * half_thickness), flow) > 0

    return neighbors[(cond1 * cond2)], cond1 * cond2


def count_around_position(positions, flow, coords_tree, half_thickness, radius=20, size=None):
    """
    The function scan along a given coordinate and count the neighbors in a column.
        position (ndarray): center of the column
        flow (ndarray): axis of the column
        coords_tree (ndarray or scipy kdtree): kdtree of the point cloud
        half_thickness (float): half thickness of the column
        radius (float): the radius of the column
        return_all (bool): True to get counts, index and

    return: counts, index, and coordinates
    """
    # convert coords to kdtree
    if not isinstance(coords_tree, spatial.kdtree.KDTree):
        coords_tree = spatial.KDTree(coords_tree)

    coords = coords_tree.data
    # get index of the neighbors. idx is an array of lists.
    indices = coords_tree.query_ball_point(positions, r=radius, workers=-1)

    coords_in_thickness = []
    column_sum = []
    idx_in_thickness = []

    # convert index to the xyz coordinate. to each array element (i.e. list), convert to coord.
    for i, idx in enumerate(indices.tolist()):
        p = positions[i, :]
        f = flow[i, :]
        neighbor_coords = coords[idx, :]
        in_thickness, subidx = isin_thickness(half_thickness, p, f, neighbor_coords)
        coords_in_thickness.append(in_thickness)
        column_sum.append(in_thickness.size)
        idx_in_thickness.append(np.asarray(idx)[subidx])

    column_sum = np.asarray(column_sum)
    if size is not None:
        column_sum = np.pad(column_sum, (0, size - column_sum.size), 'constant',
                            constant_values=(0, 0))  # pad to the size

    return column_sum, idx_in_thickness, coords_in_thickness


def count_around_position_in_disk_kernel(position, coord, half_thickness, radius, flow=None, fillna=True, batch_size=None):
    """
    Arguments
        coord (ndarray): the positions of points to be counted. First dimension is track, second is time and third is space.
        half_thickness (float): the half thickness of the disk kernel
        radius (float): the radius of the disk kernel
        flow (ndarray): orientation of the disk kernel at each point
        batch_size (int or None): This is to save RAM. Tracks with the number of batch_size will be processed at one time.
            The process will be repeated till finishing all the tracks. In case of batch_size !=None, the array should not contain np.nan.
    Return:
        ndarray:
    """

    def local_count_around_position(coords, half_thickness, radius):
        coords_tree = spatial.KDTree(coords)

        def f(arr1, arr2):
            a, _, _ = count_around_position(arr1, arr2, coords_tree, half_thickness, radius)
            return a

        return f

    def norm_1d(vector):
        return vector / np.linalg.norm(vector)

    if flow is None:
        flow = position.copy()
        flow_temp = flow.diff(dim='time')

        flow.loc[dict(time=slice(1, flow.time[-1]+1))] = flow_temp
        flow.loc[dict(time=0)] = flow.sel(time=1)
        # normalize flow
        flow = xr.apply_ufunc(
            norm_1d,
            flow,
            input_core_dims=[["space"]],
            output_core_dims=[["space"]],
            vectorize=True,
        ) # xr.apply_ufunc may be slow in this usage.
        
    if batch_size is None:
        kernel_counts = apply_function_to_array_with_array(local_count_around_position(coord, half_thickness, radius), position, flow)
        if fillna:
            kernel_counts = kernel_counts.fillna(0)
    else:
        func = local_count_around_position(coord, half_thickness, radius)
        
        end = position.shape[0]
        rn = list(range(0, end, batch_size)) + [end]
        slicers = [slice(rn[i], rn[i+1]) for i, c in enumerate(rn[:-1])]
        
        counts = []
        for slicer in tqdm(slicers):
            pos = np.vstack(position[slicer,...])
            fl = np.vstack(flow[slicer,...])
            
            counts.append(func(pos, fl).reshape(position[slicer,...].shape[:-1]))

        kernel_counts = np.vstack(counts)

    return kernel_counts


def apply_function_to_array_with_array(func, arr1, arr2, *args, **kwargs):
    """
    Arguments
        func (function): function returns flow from coordinates
        arrs (xarray): the array with shared coordinates with position
    Return:
        xarray DataArray: index of tracks, time, and space
    """
    values = arr1.copy()
    values = values.stack(pos=['time', 'track'])
    selection = ~np.isnan(values.data.T).any(axis=1)

    values2 = arr2.stack(pos=['time', 'track'])
    selection2 = ~np.isnan(values2.data.T).any(axis=1)

    selection = selection & selection2

    values_selected = values.isel(pos=selection)
    values2_selected = values2.isel(pos=selection)

    new_values = func(values_selected.data.T, values2_selected.data.T, *args, **kwargs)

    if new_values.ndim < 2:
        new_values = new_values[:, np.newaxis]

    new_values = xr.DataArray(new_values,
                              coords={'pos': values_selected.coords['pos'],
                                      'space': np.arange(new_values.shape[-1])},
                              dims=['pos', 'space']
                              )

    return new_values.unstack().T.squeeze()
