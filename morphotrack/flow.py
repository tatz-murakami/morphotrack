import numpy as np
# from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from scipy import spatial
from sklearn.preprocessing import normalize, PolynomialFeatures
from sklearn import linear_model
import ray


def align_vector_sign(vectors, guide_vector=None):
    """
    Align the sign of the vector by refering a guide vector. If the dot product of the vector and the guide vector is negative, the sign of vector is flipped.
    Highly encourage to make a guide vector before align sign.
    """
    if guide_vector is None:
        guide_vector = normalize(np.median(vectors,axis=0)[:,np.newaxis],axis=0).ravel()
    aligned_vectors = np.where(
        np.repeat(np.expand_dims(np.matmul(vectors, guide_vector) >= 0, axis=1), guide_vector.size, axis=1),
        vectors,
        -vectors
    )
    return aligned_vectors


def skeleton(binary, threshold=10):
    """
    Skeletonize the binary image using scikit-image morphology. The function removes the object smaller than threshold.
    Arguments:
        binary (ndarray):
        threshold (int): the number of pixel. The object smaller than the threshold will be removed.
    """
    skeletonized = skeletonize(binary)
    label_objects, nb_labels = ndi.label(skeletonized, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(label_objects.ravel())
    sizes[0] = 0  # To remove index=0. because it is background.
    # object_img = (label_objects == np.argmax(sizes)).astype(float)  # Get the largest objects. Making it a mask.
    sel = np.arange(sizes.size)[sizes > threshold]
    filtered_skeleton = np.isin(label_objects, sel)

    return filtered_skeleton


def guide_vector_generator(coord1, coord2):
    """
    Return the normalized vector from two coordinates.
    Arguments:
        coord1 (ndarray):
        coord2 (ndarray):
    """
    guide_vector = coord2 - coord1
    guide_vector = guide_vector / np.linalg.norm(guide_vector)

    return guide_vector


def guide_vector_generator_from_binaries(binary1, binary2):
    """
    Return the normalized vector from two binary images.
    Arguments:
        binary1 (ndarray): binary image of the layer 1 or surface of a brain
        binary2 (ndarray): binary image of white matter
    """
    coord1 = np.median(np.array(np.where(binary1)).T, axis=0)
    coord2 = np.median(np.array(np.where(binary2)).T, axis=0)

    guide_vector = guide_vector_generator(coord1, coord2)

    return guide_vector


def get_vectors_from_skeleton(binary, guide_vector, k=27, return_image=False):
    """
    Generate vectors from skeletonized binary image.
    Arguments:
         binary (ndarray): binary skeletonized image
         guide_vector (ndarray): guide vector to determine the sign of the vector
         k (int): number of neighbors to calculate the local vector.
         return_image (bool): if True, return ndarray with the shape (binary.shape + (binary.ndim,)).
            The values indicate the vectors of the skeletonized image.
    """

    positions = np.array(np.where(binary)).T
    # build kd tree to find k neighbors
    kdtree = spatial.KDTree(positions)
    mean_vectors = []

    for pos in positions.tolist():
        # Extract vectors from k-nearest neighbors.
        d, neighbors = kdtree.query(pos, k)
        neighbors = neighbors[d != 0]
        vectors_from_neighbors = normalize(positions[neighbors, :] - pos,
                                           axis=1)  # Normalize to equalize the weights
        mean_vector = np.mean(align_vector_sign(vectors_from_neighbors, guide_vector),
                              axis=0)  # Use arithmetic mean.
        mean_vector = normalize(mean_vector.reshape(-1, 1), axis=0).ravel()
        mean_vectors.append(mean_vector)

    if not return_image:
        return np.array(mean_vectors)
    else:
        vec_img = np.zeros(binary.shape + (binary.ndim,))
        vec_img[tuple(positions.T)] = mean_vectors

        return vec_img


def get_vectors_from_vessel(binary, guide_vector, threshold=10, k=27, return_image=False):
    """

    """
    print('start skeletonization')
    skeletonized = skeleton(binary, threshold=threshold)

    print('get vectors from skeletonized image')
    skeleton_vectors = get_vectors_from_skeleton(skeletonized, guide_vector, k=k)

    # Expansion of vector field to binary image.
    print('expand vectors to original image')
    skeleton_position = np.array(np.where(skeletonized)).T
    kdtree = spatial.KDTree(skeleton_position)
    positions = np.array(np.where(binary)).T
    vectors = []

    for pos in positions.tolist():
        _, nn = kdtree.query(pos, 1)  # refer nearest neighbor
        neighbor_vector = skeleton_vectors[nn]
        vectors.append(neighbor_vector)

    if not return_image:
        return np.array(vectors)
    else:
        vec_img = np.zeros(binary.shape + (binary.ndim,))
        vec_img[tuple(positions.T)] = vectors

        return np.array(vectors), vec_img


def smooth_vectors(pos, vec, radius=40, method='median'):
    """
    Smooth vectors by taking average or median of nearby vectors.
    The function avoids kdtree.query_ball_point(pos) to save RAM.
    Arguments:
        pos (ndarray):
        vec (ndarray): same shape as pos
        radius (float): smooth vectors within a radius distance
            radius 40 * 10 um = 400 um works well. diameter in real scale: 2 * radius * voxelsize.
        method: 'median' or 'mean'
    """

    @ray.remote
    def get_neighbor_vectors(position, kdtree, vectors, radius):
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        neighbors = kdtree.query_ball_point(position, radius)
        neighbor_vectors = vectors[neighbors, :]
        return neighbor_vectors

    @ray.remote
    def get_median_vector(vectors):
        # Ideally, the medoid vector should be calculated, but it is resource demanding.
        # Instead, calculate the median in each dimension and normalize to a unit vector.
        """
        vectors: ndarray
        """
        median_vector = normalize(np.median(vectors, axis=0).reshape(-1, 1), axis=0).ravel()
        return median_vector

    @ray.remote
    def get_mean_vector(vectors):
        """
        vectors: ndarray
        """
        mean_vector = normalize(np.mean(vectors, axis=0).reshape(-1, 1), axis=0).ravel()
        return mean_vector

    kdtree = spatial.KDTree(pos)

    smoothed_vectors = []

    # put large object to save memory.
    kdtree_id = ray.put(kdtree)
    vectors_id = ray.put(vec)

    if method == 'median':
        for point, point_position in enumerate(pos.tolist()):
            # Get vectors in neighbor points.
            neighbor_vectors = get_neighbor_vectors.remote(point_position, kdtree_id, vectors_id, radius)
            # Make representitive vector
            smoothed_vector = get_median_vector.remote(neighbor_vectors)
            smoothed_vectors.append(smoothed_vector)
    elif method == 'mean':
        for point, point_position in enumerate(pos.tolist()):
            # Get vectors in neighbor points.
            neighbor_vectors = get_neighbor_vectors.remote(point_position, kdtree_id, vectors_id, radius)
            # Make representitive vector
            smoothed_vector = get_mean_vector.remote(neighbor_vectors)
            smoothed_vectors.append(smoothed_vector)
    else:
        raise ValueError("method should be `median` or `mean`")

    smoothed_vectors = np.asarray(ray.get(smoothed_vectors))

    return smoothed_vectors


def remove_dissimilar_vectors(vec1, vec2, threshold='otsu'):
    dots = (vec1 * vec2).sum(axis=1) # get dot product of vectors
    if threshold == 'otsu':
        thresh = threshold_otsu(dots)
    elif isinstance(threshold_otsu, float):
        thresh = threshold_otsu
        if (thresh > 1.0) or  (thresh < -1.0):
            raise ValueError("threshold should be less than 1.0 or more than -1.0")
    else:
        raise ValueError("threshold should be `otsu` or float")
    keep = (dots >= thresh)

    return keep


def polynomial_fitting(position, values, degree=5):
    """
    Arguments:
        position (ndarray): positions of the points
        values (ndarray): target values
        degree (int): degree of polynomial. Overfitting may happen at the edge if degree is too high
    """
    poly = PolynomialFeatures(degree=degree)
    idx_ = poly.fit_transform(position)

    reg = linear_model.LinearRegression(fit_intercept=True)
    reg.fit(idx_, values)  # Fit the model
    reg.degree = degree  # save information for polynomial degree for later use

    return reg


def flow_to_normflow(reg):
    """
    The function returns the function to get normalized flow on given coordinates.
    Arguments:
        reg (linear_model.LinearRegression): the polynomial model
    Returns:
        function: function to get flow on given coordinates.
    """

    def norm_flow(coord):
        poly = PolynomialFeatures(degree=reg.degree)
        coord_ = poly.fit_transform(coord)
        normflow_on_coord = normalize(reg.predict(coord_), axis=1)
        return normflow_on_coord

    return norm_flow


