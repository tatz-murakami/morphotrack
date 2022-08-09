import numpy as np
from scipy import ndimage as ndi


def extract_largest_object_from_binary(binary_img, structure=np.ones((3, 3, 3))):
    """
    input
        binary_img: ndarray. binary image.
    return
        object_img: ndarray. binary image of the largest object
    """
    # Find object and select the largest object.
    label_objects, nb_labels = ndi.label(binary_img, structure=structure)
    sizes = np.bincount(label_objects.ravel())
    sizes[0] = 0  # To remove index=0. because it is background.
    object_img = (label_objects == np.argmax(sizes)).astype(float)  # Get the largest objects. Making it a mask.

    return object_img


def extract_largest_cluster(coord):
    """
    input
        coord: 2d array. coordinate in column.
    return
        index_array: index of points in the major cluster.
    """

    coord = coord.T.astype(int)
    binary_array = np.zeros(coord.max(axis=1) + 1)
    binary_array[tuple(coord)] = 1

    largest_object = extract_largest_object_from_binary(binary_array)
    selected_coord = np.transpose(
        largest_object.nonzero())  # Get the largest objects. Extract index. Transpose to original orientation.

    return selected_coord

