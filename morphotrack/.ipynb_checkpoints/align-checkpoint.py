import numpy as np
from scipy.ndimage import map_coordinates
from scipy import interpolate
import SimpleITK as sitk
import pydeform.sitk_api as pydeform
import morphotrack.variables
import xarray as xr
import pandas as pd


def make_displacement_map(position_arr):
    coords = np.meshgrid(*[range(x) for x in position_arr.shape], indexing='ij')
    coords = np.array(coords).astype(np.int16)
    coords[0, :, :] = position_arr

    return coords


def aligner_2d(standard, target, target2standard=None, settings=morphotrack.variables.settings):
    if target2standard is None:
        target2standard = np.meshgrid(*[range(x) for x in target.shape], indexing='ij')[0]
    displaced_tar2std = map_coordinates(target, make_displacement_map(target2standard), order=1,
                                        mode='constant')  # transform fix for better interpretation.
    fix_itk = sitk.Cast(sitk.GetImageFromArray(displaced_tar2std[np.newaxis, :, :]),
                        sitk.sitkFloat32)  # itk convert numpy zyx to xyz
    mov_itk = sitk.Cast(sitk.GetImageFromArray(standard[np.newaxis, :, :]), sitk.sitkFloat32)

    df_sitk = pydeform.register(
        fix_itk,
        mov_itk,
        settings=settings,
        num_threads=60,
        use_gpu=True
    )

    displacement = sitk.GetArrayFromImage(df_sitk)[0, :, :, 1]  # pos = sitk.GetArrayFromImage(df_sitk)[:,0,0,2]

    coords = np.meshgrid(*[range(x) for x in displacement.shape], indexing='ij')
    coords = np.array(coords).astype(np.int16)
    coords[0, :, :] = coords[0, :, :] - displacement
    adjusted_target2standard = map_coordinates(target2standard, coords, order=1, mode='constant')

    return adjusted_target2standard


def standard_generator(target, idx):
    std_track = target.sel(track=idx)
    standard = target.copy()
    standard[:] = std_track
    standard.attrs['standard_seed'] = idx

    return standard


def find_nonzero_start_end_1d(track):
    """
    Find the first and last position where non-zero values appear.
    Arguments:
        track (1darray):
    Return:
        start (int): index of first non-zero
        end (int): index of last non-zero
    """
    # identify the initial position where the counts are more than zero
    start = np.argmax(track != 0)
    # identify last position where the counts are more than zero
    end = (track.size - np.argmax(track[::-1] != 0)) - 1

    return start, end


def find_nonzero_start_end(arr, axis=0):
    """
    Find the first and last position where non-zero values appear.
    Arguments:
        arr (ndarray):
        axis (int):
    Return:
        ndarray: first values to be the start, the last values to be the end

    """
    positions = np.apply_along_axis(find_nonzero_start_end_1d, axis, arr)

    return positions


def clip(arr):
    arr[arr<0] = 0; arr[arr>arr.size-1] = arr.size-1
    return arr


def get_warp_args_1d(arr1, arr2, size=100):
    """
    Arguments:
        arr1 (ndarray):
        arr2 (ndarray):
        size (int):
    Return:
        ndarray: index to move second one to first one
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    f = interpolate.interp1d(arr1, arr2, fill_value='extrapolate')
    args = f(np.arange(size))

    # clipping
    args = clip(args)

    return args


def positional_mapping(arr1,arr2):
    return map_coordinates(arr1.T, make_displacement_map(arr2.T), order=1, mode='constant').T


def linear_shift_to_standard(counts, standard_seed):
    """
    Align the nonzero start/end position of the tracks to the nonzero start/end position of the standard track.
    Arguments:
        counts (xarray): The name of the coordinates should include track and time.
        standard_seed: the index of the standard track.
    Return:
        args
        disps
    """
    standard_start_end_pos = np.asarray(find_nonzero_start_end_1d(counts.sel(track=standard_seed).data))
    start_end_pos = find_nonzero_start_end(counts, axis=counts.get_axis_num('time'))
    size = counts.time.size

    # get positions (index) to warp.
    standard2target = np.apply_along_axis(lambda a: get_warp_args_1d(a, standard_start_end_pos, size),
                                          counts.get_axis_num('time'), start_end_pos)
    target2standard = np.apply_along_axis(lambda a: get_warp_args_1d(standard_start_end_pos, a, size),
                                          counts.get_axis_num('time'), start_end_pos)

    # the following should be cleaned up but work for now.
    standard_counts = standard_generator(counts, standard_seed).data
    displaced_std2tar = positional_mapping(standard_counts, standard2target)
    displaced_tar2std = positional_mapping(counts, target2standard)

    standard2target_xr = counts.copy();
    standard2target_xr[:] = standard2target
    target2standard_xr = counts.copy();
    target2standard_xr[:] = target2standard
    args = xr.concat([standard2target_xr, target2standard_xr], pd.Index(['s2t', 't2s'], name='displacement'))
    args.name = 'Linear_args'

    displaced_std2tar_xr = counts.copy();
    displaced_std2tar_xr[:] = displaced_std2tar
    displaced_tar2std_xr = counts.copy();
    displaced_tar2std_xr[:] = displaced_tar2std
    disps = xr.concat([displaced_std2tar_xr, displaced_tar2std_xr], pd.Index(['s2t', 't2s'], name='displacement'))
    disps.name = 'Linear_displacement'

    return args, disps


def composite_displacement(disp1st, disp2nd):
    comp_disp = positional_mapping(disp1st, disp2nd)
    return comp_disp


def warp_1d(arr, arg):
    """
    Interpolate values in arr using arg.
    Arguments
        arr (1darray):
        arg (1darray):
    Return
        arr_interp (1darray):
    """
    f = interpolate.interp1d(np.arange(arr.size), arr, fill_value='extrapolate')
    arr_interp = f(arg)

    return arr_interp


def adjust_edge(arr1, arr2):
    """
    Return arguments to move arr2 to arr1.
    Arguments
        arr1 (1darray):
        arr2 (1darray):
    Return
        arr2_warped (1darray):
        args (1darray):
    """
    pos1 = find_nonzero_start_end_1d(arr1)
    pos2 = find_nonzero_start_end_1d(arr2)
    size = arr2.size

    args = get_warp_args_1d(pos1, pos2, size)
    arr2_warped = warp_1d(arr2, args)

    return arr2_warped, args


def composite_displacement_1d(arg1, arg2):
    """
    Interpolate values in arr using arg.
    Arguments
        arg1 (1darray): arguments for the first warp
        arg2 (1darray): arguments for the second warp
    Return
        arg_interp (1darray):
    """
    f = interpolate.interp1d(np.arange(arg1.size), arg1, fill_value='extrapolate')
    arg_composite = f(arg2)

    return arg_composite



