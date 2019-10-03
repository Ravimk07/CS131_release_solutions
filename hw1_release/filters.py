"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    if kernel.shape[0] > image.shape[0] or kernel.shape[1] > image.shape[1]:
    	print('Kernel is bigger than image! Provide arguments in swap order.')
    	return image

    flipped = kernel[::-1, ::-1]
    # skip edges as there was no information on which padding should be used
    row_moves = kernel.shape[0] // 2
    column_moves = kernel.shape[1] // 2
    for r in range(row_moves, image.shape[0] - row_moves):
    	for c in range(column_moves, image.shape[1] - column_moves):
    		sum_v = 0
    		for i in range(-row_moves, row_moves + 1):
    			for j in range(-column_moves, column_moves + 1):
    				sum_v += image[r + i, c + j] * flipped[i + row_moves, j + column_moves]
    		out[r, c] = sum_v

    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    out[pad_height : H + pad_height, pad_width : W + pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    row_pad = kernel.shape[0] // 2
    column_pad = kernel.shape[1] // 2
    padded = zero_pad(image, row_pad, column_pad)
    flipped = kernel[::-1, ::-1]

    for r in range(row_pad, padded.shape[0] - row_pad):
    	for c in range(column_pad, padded.shape[1] - column_pad):
    		out[r - row_pad, c - column_pad] = np.sum(padded[r - row_pad : r + row_pad + 1,
    		 c - column_pad : c + column_pad + 1] * flipped)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    fft_size = [Hi+Hk-1,Wi+Wk-1]
    fft_img = np.fft.fft2(image,fft_size)
    fft_ker = np.fft.fft2(kernel,fft_size)
    conv_img = np.fft.ifft2(fft_img*fft_ker).real
    out = conv_img[Hk//2:conv_img.shape[0]-Hk//2,Wk//2:conv_img.shape[1]-Wk//2]
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    flipped_g = g[::-1, ::-1]
    ### YOUR CODE HERE
    out = conv_fast(f, flipped_g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_m = g - g.mean()
    out = cross_correlation(f, g_m)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    out = np.zeros(f.shape)
    row_pad = g.shape[0] // 2
    column_pad = g.shape[1] // 2
    padded = zero_pad(f, row_pad, column_pad)
    kernel = (g - g.mean()) / g.std()

    for r in range(row_pad, padded.shape[0] - row_pad):
    	for c in range(column_pad, padded.shape[1] - column_pad):
    		patch = padded[r - row_pad : r + row_pad + 1, c - column_pad : c + column_pad + 1]
    		patch = (patch - patch.mean())/ patch.std()
    		out[r - row_pad, c - column_pad] = np.sum(patch * kernel)
    ### END YOUR CODE

    return out
