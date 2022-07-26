from scipy.integrate import solve_ivp
from scipy import spatial
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mutual_info_score
import ray
import xarray as xr
import morphotrack.points



def mutual_information(s1, s2, bins=10):
    """
    #https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    """
    hist = np.histogram2d(s1, s2, bins)[0]
    mi = mutual_info_score(None,None,contingency=hist)
    return mi


def cosine_similarity(a1, a2):
    return np.dot(a1, a2) / (np.linalg.norm(a1)*np.linalg.norm(a2))


def moving_average(x, w=10):
    return np.convolve(x, np.ones(w), 'same') / w


def polynomial_eq(y, coeff, ij_powers, intercept):
    x0, x1, x2 = y
    return sum(
        (b * (x0 ** ij_powers[i, 0]) * (x1 ** ij_powers[i, 1]) * (x2 ** ij_powers[i, 2]) for i, b in enumerate(coeff))
    ) + intercept


def polynomial_vectorfield_generator(reg):
    """
    Arguments:
        reg : scikit-learn Linear regression object
    Return:
        function: ready to use for scipy solve ivp
    """
    coeff = reg.coef_; degree = reg.degree
    poly = PolynomialFeatures(degree=degree)
    poly.fit(np.ones((1, coeff.shape[0])))  # this is just to get poly.powers_
    ij_powers = poly.powers_
    intercept = reg.intercept_
    if not isinstance(intercept, (np.ndarray, np.generic)):
        intercept = np.zeros(coeff.shape[0])

    def polynomial_vectorfield(t, y):
        # x0, x1, x2 = w
        # Create f = (x0', x1', x2')
        # f = [
        #     polynomial_eq(y, list(coeff[0]), ij_powers, intercept[0]),
        #     polynomial_eq(y, list(coeff[1]), ij_powers, intercept[1]),
        #     polynomial_eq(y, list(coeff[2]), ij_powers, intercept[2])
        # ]
        f = [polynomial_eq(y, list(coeff[i]), ij_powers, intercept[i]) for i in range(coeff.shape[0])]

        return f

    return polynomial_vectorfield


def fetch_value_in_range(array, index_array, return_inloc=False):
    """
    The function fetch the value in a given array using indexing array.
    The indexing array can have the index outside of the given array. The outside index will be ignored.
    Arguments:
        array (ndarray): numpy array from where the values to be fetched.
        index_array (ndarray): numpy array for indexing. Each row has the position of the array to be fetched.
        return_inloc (bool): True to return the position where index was in array.
    """
    dim = array.ndim
    index_array = index_array.T

    index_array[index_array < 0] = 0
    inloc = np.any(index_array >= 0, axis=0)
    for n in range(dim):
        upper_inloc = (index_array[n, :] < (array.shape[n] - 0.5))
        index_array[n, :] = np.where(
            upper_inloc,
            index_array[n, :],
            array.shape[n] - 1
        )
        inloc = np.logical_and(inloc, upper_inloc)
    index_array = np.around(index_array).astype(
        int)  # Ideally, interpolating is better than rounding. Interpolation is too intensive.
    values = array[tuple([index_array[i, :] for i in range(dim)])]

    if return_inloc:
        return values, inloc
    else:
        return values


def measure_travelled_distance(array):  # Cumulative distance. Distance from point t0 to end point.
    """
    This function returns how far travelled from the first row to the last row.
    Arguments:
        array: Numpy array. Each row is spacial coordinate.
    Return:
        ndarray: travelled distance
    """
    increment_distance = np.linalg.norm(
        array - np.roll(array, 1, axis=0),
        2,
        axis=1
    )
    increment_distance[0] = 0
    travelled_distance = np.cumsum(increment_distance)

    return travelled_distance


def fetch_value_in_position(xarr, arr):
    """

    """
    def fetch_value_in_array(array):
        def f(index):
            a = morphotrack.track.fetch_value_in_range(array, index, return_inloc=False)
            return a

        return f

    return apply_function_to_position(fetch_value_in_array(arr), xarr)


def find_region(xarr, arr, extract_largest=False):
    """
    Arguments:
        xarr (xarray);
        arr (ndarray): binary image of the segmented region (e.g. segmented white matter)
        extract_largest (bool): if True, extract only the largest segment
    """
    region_pos = fetch_value_in_position(xarr, arr)
    region_pos = region_pos.astype(float).fillna(0)
    if extract_largest:
        region_pos.data = np.apply_along_axis(
            lambda a: morphotrack.binary.extract_largest_object_from_binary(a, np.ones(3)), axis=1, arr=region_pos.data)

    return region_pos


def apply_function_to_position(func, arr1, *args, **kwargs):
    """
    Arguments
        func (function): function returns flow from coordinates
    Return:
        xarray DataArray: index of tracks, time, and space
    """
    values = arr1.copy()
    values = values.stack(pos=['time', 'track'])
    selection = ~np.isnan(values.data.T).any(axis=1)
    values_selected = values.isel(pos=selection)
    new_values = func(values_selected.data.T, *args, **kwargs)

    if new_values.ndim < 2:
        new_values = new_values[:, np.newaxis]

    new_values = xr.DataArray(new_values,
                              coords={'pos': values_selected.coords['pos'],
                                      'space': np.arange(new_values.shape[-1])},
                              dims=['pos', 'space']
                              )

    return new_values.unstack().T.squeeze()


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


class FieldTracker:
    def __init__(self, model, seeds=None):
        """
        Arguments:
            model (function): the function is passed to solve_ivp
            seeds (ndarray): the coordinates. Nx3 array.
        """
        self.model = model  # model is to return vectors from coordinates.

        # sort seeds using isomap
        isomap1d = morphotrack.points.isomap_wrapper(seeds, n_components=1)
        sort_args = np.argsort(isomap1d, axis=0)

        self.sort_args = sort_args

        self.seeds = seeds[sort_args, :]
        self.seeds_xr = xr.DataArray(self.seeds,
                                     coords={'track': np.arange(self.seeds.shape[0]),
                                             'space': np.arange(self.seeds.shape[-1])},
                                     dims=['track', 'space']
                                     )

        self.t_positions = None

    def solve_ode(self, t_start, t_end):
        """
        Arguments:
            t_start (int):
            t_end (int):
        return:
            # xarray DataArray: index of tracks, time, and space
        """

        @ray.remote
        def par_solve_ode(coord):
            ode_solution = solve_ivp(
                self.model,
                (t_start, t_end),
                y0=np.asarray(coord),
                t_eval=np.linspace(t_start, t_end - 1, t_end),
            )

            return ode_solution

        ode_sols = []
        for seed in self.seeds:
            ode_sols.append(par_solve_ode.remote(seed))
        ode_sols = ray.get(ode_sols)

        # return ode_sols
        t_position = []
        for sol in ode_sols:
            # pad the np.nan to the size
            s = sol.y.T.shape[0]
            t_position.append(np.pad(sol.y.T, ((0, t_end - s), (0, 0)), 'constant', constant_values=np.nan))

        t_position = np.stack(t_position)
        t_position = xr.DataArray(t_position,
                                  coords={'track': np.arange(t_position.shape[0]),
                                          'time': np.arange(t_start, t_end),
                                          'space': np.arange(t_position.shape[-1])},
                                  dims=['track', 'time', 'space']
                                  )

        self.t_positions = t_position

    def apply_function_to_position(self, func, *args, **kwargs):
        """
        Arguments
            func (function): function returns flow from coordinates
        Return:
            xarray DataArray: index of tracks, time, and space
        """
        values = self.t_positions.copy()
        values = values.stack(pos=['time', 'track'])
        selection = ~np.isnan(values.data.T).any(axis=1)
        values_selected = values.isel(pos=selection)
        new_values = func(values_selected.data.T, *args, **kwargs)

        if new_values.ndim < 2:
            new_values = new_values[:, np.newaxis]

        new_values = xr.DataArray(new_values,
                                  coords={'pos': values_selected.coords['pos'],
                                          'space': np.arange(new_values.shape[-1])},
                                  dims=['pos', 'space']
                                  )

        return new_values.unstack().T.squeeze()

    def apply_function_to_position_with_array(self, func, arr2, *args, **kwargs):
        """
        Arguments
            func (function): function returns flow from coordinates
            arrs (xarray): the array with shared coordinates with position
        Return:
            xarray DataArray: index of tracks, time, and space
        """
        values = self.t_positions.copy()
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

    def count_around_position_in_disk_kernel(self, coord, half_thickness, radius, flow=None, fillna=True):
        """
        Arguments
            coord (ndarray): the positions of points to be counted
            half_thickness (float): the half thickness of the disk kernel
            radius (float): the radius of the disk kernel
            flow (ndarray): orientation of the disk kernel at each point
        Return:
            ndarray:
        """

        def count_around_position(coords, half_thickness, radius):
            coords_tree = spatial.KDTree(coords)

            def f(arr1, arr2):
                a, _, _ = morphotrack.points.count_around_position(arr1, arr2, coords_tree, half_thickness, radius)
                return a

            return f

        def norm_1d(vector):
            return vector / np.linalg.norm(vector)

        if flow is None:
            flow = self.t_positions.copy()
            flow_temp = flow.diff(dim='time')

            flow[dict(time=slice(1, None))] = flow_temp
            flow[dict(time=0)] = flow.sel(time=1)
            # normalize flow
            flow = xr.apply_ufunc(
                norm_1d,
                flow,
                input_core_dims=[["space"]],
                output_core_dims=[["space"]],
                vectorize=True,
            ) # xr.apply_ufunc may be slow in this usage.

        kernel_counts = self.apply_function_to_position_with_array(count_around_position(coord, half_thickness, radius), flow)
        if fillna:
            kernel_counts = kernel_counts.fillna(0)

        return kernel_counts

    def fetch_value_in_position(self, arr):
        """
        Arguments
            arr (ndarray): array with values.
        Return
            ndarray:
        """

        def fetch_value_in_array(array):
            def f(index):
                a = morphotrack.track.fetch_value_in_range(array, index, return_inloc=False)
                return a

            return f

        return self.apply_function_to_position(fetch_value_in_array(arr))
