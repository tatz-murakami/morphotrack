import morphotrack.variables
import morphotrack.align
import numpy as np
import SimpleITK as sitk
import pydeform.sitk_api as pydeform
from tqdm import tqdm


def tracks_aligner(fix, mov, clipping=True, settings=morphotrack.variables.settings, num_threads=-1, use_gpu=True,
                   **kwargs):
    """
    Wrapping deform (https://github.com/simeks/deform). Assume 1d arrays as inputs
    Arguments:
        fix (1darray): fixed target.
        mov (1darray): moving object.
        clipping (bool):
        settings: settings for pydeform
        num_threads (int):
        use_gpu (bool):s
    Return:
        mov2fix_args: the index to warp mov to fix
        fix2mov_args: the index to warp fix to mov. Please note deform uses non symmetric transformation.
    """

    fix_itk = sitk.Cast(sitk.GetImageFromArray(fix[:, np.newaxis, np.newaxis]), sitk.sitkFloat32)
    mov_itk = sitk.Cast(sitk.GetImageFromArray(mov[:, np.newaxis, np.newaxis]), sitk.sitkFloat32)
    df_sitk = pydeform.register(
        fix_itk,
        mov_itk,
        settings=morphotrack.variables.settings,
        num_threads=num_threads,
        use_gpu=use_gpu,
        **kwargs
    )
    pos = sitk.GetArrayFromImage(df_sitk)[:, 0, 0, 2]
    mov2fix_args = np.arange(pos.size) + pos
    fix2mov_args = np.arange(pos.size) - pos

    if clipping:
        mov2fix_args = morphotrack.align.clip(mov2fix_args)
        fix2mov_args = morphotrack.align.clip(fix2mov_args)

    return mov2fix_args, fix2mov_args


def track_wise_aligner(fix2d, mov2d, **kwargs):
    """
    Wrapping deform (https://github.com/simeks/deform). Assume 2d arrays as inputs. Each row is a track.
    Arguments:
        fix2d (2darray): fixed target.
        mov2d (2darray): moving object.
    Return:
        args: the index to warp mov to fix
    """
    mov2fix_args = np.zeros(fix2d.shape, dtype=float)
    fix2mov_args = np.zeros(fix2d.shape, dtype=float)

    for l in tqdm(np.arange(fix2d.shape[0])):
        mov2fix_args[l, :], fix2mov_args[l, :] = tracks_aligner(fix2d[l, :], mov2d[l, :], **kwargs)

    return mov2fix_args, fix2mov_args

