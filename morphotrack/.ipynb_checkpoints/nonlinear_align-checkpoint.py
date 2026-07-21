import morphotrack.variables
import morphotrack.align
import morphotrack.dtw
import numpy as np
import SimpleITK as sitk
import pydeform.sitk_api as pydeform
from tqdm import tqdm


def tracks_aligner(fix, mov, clipping=True, fft_threshold=None, settings=morphotrack.variables.settings, num_threads=-1, use_gpu=True,
                   **kwargs):
    """
    Wrapping deform (https://github.com/simeks/deform). Assume 1d arrays as inputs
    Arguments:
        fix (1darray): fixed target.
        mov (1darray): moving object.
        clipping (bool):
        fft_threshold (float): recommended value, 1e3
        settings: settings for pydeform
        num_threads (int):
        use_gpu (bool):
    Return:
        mov2fix_args: the index to warp mov to fix
        fix2mov_args: the index to warp fix to mov. Please note deform uses non symmetric transformation.
    """
    if fft_threshold is not None:
        fix = morphotrack.dtw.filter_signal(fix, threshold=fft_threshold)
        fix[fix == 0] = 0
        mov = morphotrack.dtw.filter_signal(mov, threshold=fft_threshold)
        mov[mov == 0] = 0

    fix_itk = sitk.Cast(sitk.GetImageFromArray(fix[:, np.newaxis, np.newaxis]), sitk.sitkFloat32)
    mov_itk = sitk.Cast(sitk.GetImageFromArray(mov[:, np.newaxis, np.newaxis]), sitk.sitkFloat32)
    df_sitk = pydeform.register(
        fix_itk,
        mov_itk,
        settings=settings,
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


def non_linear_align_1d(arr1, arr2, pre_linear_alignment=True):
    """
    Non-linear alignment of arr1 and arr2.
    Arguments
        arr1 (1darray):
        arr2 (1darray):
    Return
        arr2_nonlinear_warped (1darray):
        arg_composite (1darray): arguments to warp arr2 to arr2_nonlinear_warped
    """
    if pre_linear_alignment:
        arr2_warped, arg0 = morphotrack.align.adjust_edge(arr1, arr2)
    else:
        arr2_warped = arr2
        arg0 = np.arange(arr2.size)
    arg1, _ = morphotrack.nonlinear_align.tracks_aligner(arr1, arr2_warped)
    arg_composite = morphotrack.align.composite_displacement_1d(arg0, arg1)
    arr2_nonlinear_warped = morphotrack.align.warp_1d(arr2, arg_composite)

    return arr2_nonlinear_warped, arg_composite


def filter_signal(signal, threshold=1e3):
    fourier = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0

    fft_filtered_signal = np.fft.irfft(fourier)
    # pad to original size
    if fft_filtered_signal.size < signal.size:
        fft_filtered_signal = np.append(fft_filtered_signal,[fft_filtered_signal[-1]]*(signal.size-fft_filtered_signal.size))

    return fft_filtered_signal

