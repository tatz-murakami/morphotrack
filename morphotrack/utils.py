import numpy as np


def fourier_masker(image, val, depth=25, height=25, width=25):
    """
    Sharp filtering in fourier space for 3D
    """
    offset = image.min()
    image = image - offset
    image_fourier = np.fft.fftshift(np.fft.fftn(image))

    # leave the center of the images
    cropping_region = (depth, height, width)
    image_transformed = np.full_like(image_fourier, val)
    slicing = tuple(slice(round(image_fourier.shape[i] / 2) - cropping_region[i],
                          round(image_fourier.shape[i] / 2) + cropping_region[i]) for i in range(len(cropping_region)))
    image_transformed[slicing] = image_fourier[slicing]

    return abs(np.fft.ifftn(image_transformed)) + offset


def window3d(w):
    # Convert a 1D filtering kernel to 3D
    # eg, window3D(numpy.hamming(5))
    L = w.shape[0]
    m1 = np.outer(np.ravel(w), np.ravel(w))
    win1 = np.tile(m1, np.hstack([L,1,1]))
    m2 = np.outer(np.ravel(w), np.ones([1,L]))
    win2 = np.tile(m2, np.hstack([L,1,1]))
    win2 = np.transpose(win2, np.hstack([1,2,0]))
    win = np.multiply(win1, win2)

    return win


def window2d(w):
    # Convert a 1D filtering kernel to 3D
    # eg, window3D(numpy.hamming(5))
    win = np.sqrt(np.outer(w, w))

    return win


def fourier_masker_hamming(image, val=1, size=50):
    """
    Gradual filtering in fourier space for 3D
    size should be an even number.
    """
    # size should be even number
    offset = image.min()
    image = image - offset
    image_fourier = np.fft.fftshift(np.fft.fftn(image))

    # get dimension
    dim = image.ndim

    # leave the center of the images
    image_transformed = np.full_like(image_fourier, val)
    weight = np.zeros_like(image_fourier)
    if dim == 2:
        hamming = window2d(np.hamming(size))
    elif dim == 3:
        hamming = window3d(np.hamming(size))

    slicing = tuple(slice(round(i / 2 - size / 2), round(i / 2 + size / 2)) for i in image_fourier.shape)
    weight[slicing] = hamming
    image_transformed[slicing] = (image_fourier * weight)[slicing]

    return abs(np.fft.ifftn(image_transformed)) + offset
