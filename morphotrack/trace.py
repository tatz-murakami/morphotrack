from scipy.integrate import solve_ivp
from scipy import interpolate
import numpy as np
import pandas as pd


def fetch_value_in_range(array, index_array):
    """
    The function fetch the value in a given array using indexing array.
    The indexing array can have the index outside of the given array. The outside index will be ignored.
    array: numpy array to be fetched.
    index_array: numpy array for indexing. Each column has the position of the array to be fetched.
    """
    dim = array.ndim
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
    return values, inloc


def measure_travelled_distance(array):  # Cumulative distance. Distance from point t0 to end point.
    """
    array: Numpy array. Each row is spacial coordinate.
    This function returns how far travelled from the first row to the last row.
    """
    increment_distance = np.linalg.norm(
        array - np.roll(array, 1, axis=0),
        2,
        axis=1
    )
    increment_distance[0] = 0
    travelled_distance = np.cumsum(increment_distance)
    return travelled_distance


def count_inloc_outloc(loc, inloc_outloc_count=[]):  # set inloc_outloc_count=[] in the argument to prevent a bug.
    """
    The function counts the number of the continous True and False.
    input:
        loc: list of boolean.
    return:
        inloc_outloc_count: the odd elements are the counts of False, the even elements are the counts of True.
    """
    # first element is always the number of inloc.
    count = 0
    while not loc[0]:
        loc.pop(0)
        count += 1
        if not loc:
            inloc_outloc_count.append(count)
            break
    else:
        inloc_outloc_count.append(count)
        invert_loc = [not x for x in loc]
        inloc_outloc_count = count_inloc_outloc(invert_loc, inloc_outloc_count)

    return inloc_outloc_count


class Result(dict):
    """
    This class is a subclass of dict with attribute accessors.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys())


class TrajectoryAnalysisResult(Result):
    """
    Attributes
    ----------
    t_start: int
        Time to start the ode_ivp.
    t_end: int
        Time to end the ode_ivp.
    ode_solution: object
        Returned object of scipy solve_ivp.
    distance: 1d numpy array
        Distance travelled from t_start to t_end.
    values_on_trajectory: 1d numpy array
        Values on the trajectory.
    inloc: boolean 1d numpy array
        The trajectory is inside of mask or not. True if yes.
    inloc_outloc_count: list

    continuity: boolean
        True if the trajectory is continuous, false if fragmented.
    """
    def __init__(self):
        self.t_start = None
        self.t_end = None
        self.ode_solution = None
        self.distance = None
        self.values_on_trajectory = None
        self.inloc = None
        self.inloc_outloc_count = None
        self.continuity = True
        self.warp_position = None


class DtwAlignmentResult(Result):
    """
    Attributes
    ----------
    dtw_coverages: Pandas Series
        The coverage of open-end DTW if the object is used as a query.
    #cluster_label: int
    #    Cluster label of hierarchical clustering.
    dtw_selection: boolean
        If the object is included in the major cluster or not.
    """
    def __init__(self):
        self.dtw_coverages = pd.Series([], dtype=float)
        self.mutual_information = pd.Series([], dtype=float)
        # self.cluster_label = None
        self.dtw_selection = False


class VirtualRG:
    def __init__(self, seed_position, model, *model_args):
        self.seed_number = None
        self.seed_position = seed_position
        self.model = model
        self.model_args = model_args
        self.trajectory = TrajectoryAnalysisResult()
        self.dtw = DtwAlignmentResult()
        # self.dtw_coverages = pd.Series([], dtype=float)

    def set_seed_number(self, seed_number):
        self.seed_number = seed_number
        self.dtw.dtw_coverages = self.dtw.dtw_coverages.rename(self.seed_number)
        # self.dtw.dtw_coverages[self.seed_number] = 1.0  # Self alignment is always 1.0
        self.dtw.mutual_information = self.dtw.mutual_information.rename(self.seed_number)

    def set_model_parameters(self, *model_args):
        self.model_args = model_args

    def solve_ode(self, t_start, t_end):  # Wrapping scipy solve_ivp
        self.trajectory.t_start = t_start
        self.trajectory.t_end = t_end
        ode_solution = solve_ivp(
            self.model,
            (t_start, t_end),
            y0=self.seed_position,
            t_eval=np.linspace(t_start, t_end, t_end + 1),
            args=self.model_args
        )
        self.trajectory.ode_solution = ode_solution
        self.trajectory.distance = measure_travelled_distance(ode_solution.y.T)

    def fetch_value(self, array, mask=None):
        self.trajectory.values_on_trajectory, self.trajectory.inloc = fetch_value_in_range(
            array,
            self.trajectory.ode_solution.y
        )
        if not (mask is None):
            tissue_inloc, _ = fetch_value_in_range(
                mask,
                self.trajectory.ode_solution.y
            )
            self.trajectory.inloc = np.logical_and(self.trajectory.inloc, tissue_inloc)

        inloc_outloc_count = count_inloc_outloc(self.trajectory.inloc.tolist(), inloc_outloc_count=[])
        self.trajectory.inloc_outloc_count = inloc_outloc_count
        self.trajectory.continuity = (len(inloc_outloc_count) == 2 or len(inloc_outloc_count) == 3)

    def update_coverage(self, pair_seed_num, coverage):
        self.dtw.dtw_coverages[pair_seed_num] = coverage

    def update_mutual_information(self, pair_seed_num, mutual_info):
        self.dtw.mutual_information[pair_seed_num] = mutual_info
