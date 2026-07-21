import numpy as np
from scipy import interpolate
import xarray as xr


def position2distance(positions):
    """
    positions: 2d array. Nx3.
    """
    diff = np.diff(positions, axis=0, prepend=0)  # prepend to make the size consistent
    dist = np.linalg.norm(diff, 2, axis=1)
    dist[0] = 0
    return dist


def position2distance_xr(arr):
    """
    arr (xarray)
    """
    dist = []
    dist_xr = arr.copy().isel(space=0).drop_vars('space')
    for i in arr.track.data:
        dist.append(position2distance(arr.sel(track=i)))

    dist_xr[:] = np.stack(dist)
    return dist_xr


def map2standardspace_1d(coordinate, standard_coord_val):
    """
    coordinate (1d array):
    standard_coord_val (1d array):
    """
    f = interpolate.interp1d(np.arange(standard_coord_val.size), standard_coord_val, fill_value="extrapolate", kind='linear')
    return f(coordinate)


def map2standard_vectorspace(coordinate, standard_coord_val):
    """
    coordinate (1d array):
    standard_coord_val (2d array):
    """
    return np.apply_along_axis(lambda a: map2standardspace_1d(coordinate, a), 0, standard_coord_val)


def map2standard_vectorspace_xr(coordinate, standard_coord_val):
    """
    coordinate (xarray):
    standard_coord_val (xarray):
    """
    vecs = []

    for i in coordinate.track.data:
        vecs.append(map2standard_vectorspace(coordinate.sel(track=i), standard_coord_val))

    arr = xr.DataArray(np.stack(vecs),
                       coords={'track': coordinate.track,
                               'time': coordinate.time,
                               'space': standard_coord_val.space},
                       dims=['track', 'time', 'space']
                       )

    return arr


def logFC_distance_t2s(t_xr, s_xr, window=10):
    logfc = t_xr.rolling(time=window, center=True).mean() / s_xr.rolling(time=window, center=True).mean()
    logfc = xr.where(logfc == np.inf, np.nan, logfc)
    logfc = xr.where(logfc == -np.inf, np.nan, logfc)
    logfc = np.log2(logfc)
    return logfc

