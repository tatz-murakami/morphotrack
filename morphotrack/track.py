from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import ray
import xarray as xr


def polynomial_eq(w, coeff, ij_powers):
    x0, x1, x2 = w
    return sum(
        (b * (x0 ** ij_powers[i, 0]) * (x1 ** ij_powers[i, 1]) * (x2 ** ij_powers[i, 2]) for i, b in enumerate(coeff)))


def polynomial_vectorfield_generator(coeff, degree):
    """
    Arguments:
        coeff : coefficient of the polynomial fitting
        degree: the degree of the polynomial fitting
    Return:
        function: ready to use for scipy solve ivp
    """
    poly = PolynomialFeatures(degree=degree)
    poly.fit(np.ones((1, coeff.shape[0])))  # this is just to get poly.powers_
    ij_powers = poly.powers_

    def polynomial_vectorfield(t, w):
        x0, x1, x2 = w
        # Create f = (x0', x1', x2')
        f = [
            polynomial_eq(w, list(coeff[0]), ij_powers),
            polynomial_eq(w, list(coeff[1]), ij_powers),
            polynomial_eq(w, list(coeff[2]), ij_powers)
        ]
        return f

    return polynomial_vectorfield


def fetch_value_in_range(array, index_array, return_inloc=True):
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


class FieldTracker:
    def __init__(self, model, points=None):
        """
        Arguments:
            model (function): the function is passed to solve_ivp
            points (ndarray): the coordinates. Nx3 array.
        """
        self.model = model  # model is to return vectors from coordinates.
        self.points = points
        # self.ode_solution = None
        self.points_xr = xr.DataArray(points,
                                      coords={'track': np.arange(points.shape[0]),
                                              'space': np.arange(points.shape[-1])},
                                      dims=['track', 'space']
                                      )
        self.t_positions = None

    def solve_ode(self, t_start, t_end):
        """
        Arguments:
            t_start (int):
            t_end (int):
        return:
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
        for point in self.points:
            ode_sols.append(par_solve_ode.remote(point))
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

    def continuity(self, binary_img):
        """
        Arguments:
            binary_img (ndarray): the binary image
        return:
        """
