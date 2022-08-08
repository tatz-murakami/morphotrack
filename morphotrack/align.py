import numpy as np
from scipy.ndimage import map_coordinates
import SimpleITK as sitk
import pydeform.sitk_api as pydeform
import morphotrack.variables


def make_displacement_map(position_arr):
    coords = np.meshgrid(*[range(x) for x in position_arr.shape], indexing='ij')
    coords = np.array(coords).astype(np.int16)
    coords[0, :, :] = position_arr

    return coords


def aligner(standard, target, target2standard=None, settings=morphotrack.variables.settings):
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
    std_track = target.isel(track=idx)
    standard = target.copy()
    standard[:] = std_track

    return standard

