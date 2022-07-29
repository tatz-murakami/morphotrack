from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import normalize
import numpy as np
from scipy import spatial
from sklearn import manifold


def get_norm_flow_on_coordinate(degree, clf):
    """
    The function returns the function to get normalized flow on given coordinates.
    Arguments:
        degree (int): the degree of polynomial
        clf (linear_model.LinearRegression): the polynomial model
    Returns:
        function: function to get flow on given coordinates.
    """
    def get_flow(coord):
        poly = PolynomialFeatures(degree=degree)
        coord_ = poly.fit_transform(coord)
        flow_on_coord = normalize(clf.predict(coord_), axis=1)
        return flow_on_coord

    return get_flow


def isin_thickness(half_thickness, position, flow, neighbors):
    """
    The function returns the neighbors within a certain thickness.
        half_thickness (float): half thickness of the column
        position (ndarray): center of the column
        flow (ndarray): axis of the column
        neighbors (ndarray): Nx3 array
    return: neighbors in thickness, boolean for indexing
    """
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
        neighbor_coords = coords[idx, :]
        in_thickness, subidx = isin_thickness(half_thickness, positions[i, :], flow[i, :], neighbor_coords)
        coords_in_thickness.append(in_thickness)
        column_sum.append(in_thickness.size)
        idx_in_thickness.append(np.asarray(idx)[subidx])

    column_sum = np.asarray(column_sum)
    if size is not None:
        column_sum = np.pad(column_sum, (0, size - column_sum.size), 'constant',
                            constant_values=(0, 0))  # pad to the size

    return column_sum, idx_in_thickness, coords_in_thickness


def grid_pos_in_circle(radius, sigma=5.0):
    """
    radius controls the size
    sigma controls the number of the point
    """
    positions = []

    X = int(sigma)
    for i in range(-X, X + 1):
        Y = int(pow(sigma * sigma - i * i, 1/2))
        for j in range(-Y, Y + 1):
            positions.append((i, j))
    grid_in_circle = np.asarray(positions)
    grid_in_circle = grid_in_circle * radius / sigma

    return grid_in_circle


def grid_pos_in_circle_3d(radius, sigma=5.0, zero_axis=0):
    """
    """
    positions = grid_pos_in_circle(radius, sigma=sigma)
    return np.asarray(np.insert(positions,[zero_axis],np.zeros(positions.shape[0])[:,np.newaxis], axis=1))


def pick_median_nearest_point(array, k, median_shift=False):
    """
    Find points which are within a certain distance (r) to the median point. If median_shift is true, the point closest to the median is used as a center of kNN.
    Arguments:
        array (ndarray): 3D coordinate of points.
        r (int):  number of neighbors
        median_shift (boolean):
    Return:
        index: index (row) of the nearest neighbor
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    kd_tree = spatial.KDTree(array)
    median = np.median(array, axis=0)
    if median_shift:
        _, i = kd_tree.query(median, 1)
        median_nn = array[i, :]
        _, index = kd_tree.query(median, k)
    else:
        _, index = kd_tree.query(median, k)

    return index


def isomap_wrapper(coordinate, n_neighbors=20, n_components=1, **kwargs):
    """
    """
    iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1, **kwargs)
    iso.fit(coordinate)
    transformed_coordinate = iso.transform(coordinate).squeeze()

    return transformed_coordinate


def position_on_circunference(radius, n_points=8, dim=3):
    """
    n_points control the number of the points on circunference.
    """
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    position = np.stack((y, x))
    if dim == 3:
        position = np.vstack((np.zeros_like(x), position))

    return position.T


def get_orthonogals(k, x=np.asarray([0.0, 1.0, 0.0])):
    """
    k: 2darray. nx3
    Assume input vector k to be normalized
    """
    x = x - k.dot(x)[:, None] * k  # make x orthogonal to k
    x = x / np.linalg.norm(x, axis=1)[:, None]  # normalize it

    y = np.cross(k, x)

    return k, x, y


def rotate_with_normals(coord, vectors):
    """
    rotate agianst normal. (0,0,0) is used as the origin of the rotation
        coord: 2darray. ix3. coordinates to be rotated.
        vectors: 2darray. jx3. j is a number of the normal vectors.
    return
        points: 3darray. ixjx3. rotated position of coordinates for each normal vector.
    """
    M = get_orthonogals(vectors)

    points = np.einsum('ij,jkl->ikl', coord, np.asarray(M)) # [measurement point, position in line, xyz coordinate]

    return points


def get_local_flux(positions, vector_field, radius, dim=3, n_points=8):
    """
    positions (ndarray):
    vector_field (function): return vector for each position
    radius (int):
    """
    flow = vector_field(positions)
    # generate measuring points. circles around position
    points_on_cicle = position_on_circunference(radius, n_points=n_points, dim=dim)
    rotated_points = rotate_with_normals(points_on_cicle, flow)
    measuring_point = rotated_points + positions

    # calculate local flux
    temp = vector_field(measuring_point.reshape(-1, measuring_point.shape[-1]))
    vectors_on_points = temp.reshape(measuring_point.shape)  # get vectors on measuring points
    local_flux = np.einsum('ijk,ijk->j', vectors_on_points, rotated_points / radius) / points_on_cicle.shape[0]

    return local_flux

