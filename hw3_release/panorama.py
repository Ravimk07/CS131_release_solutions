"""
CS131 - Computer Vision: Foundations and Applications
Assignment 3
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/27/2018
Python Version: 3.5+
"""

import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above.

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    d2x = dx * dx
    d2y = dy * dy
    dxy = dx * dy

    ### YOUR CODE HERE
    w_pad = window_size // 2
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            M = np.zeros((2, 2))
            for i in range(-w_pad, w_pad + 1):
                for j in range(-w_pad, w_pad + 1):
                    ix = r + i
                    iy = c + j
                    M += window[i + w_pad, j + w_pad] * np.array([[d2x[ix, iy], dxy[ix, iy]], 
                        [dxy[ix, iy], d2y[ix, iy]]])
            response[r, c] = np.linalg.det(M) - k * (np.trace(M) ** 2)
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        If a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    feature = []
    ### YOUR CODE HERE
    den = patch.std()
    if den == 0:
        den = 1
    normalized = (patch - patch.mean()) / den
    feature = normalized.flatten()
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    a_m = []
    for i1 in range(desc1.shape[0]):
        dists_row = dists[i1].copy()
        i2 = np.argmin(dists_row)
        dists_row[i2] = dists_row.max()
        i3 = np.argmin(dists_row)

        if dists[i1, i2] / dists[i1, i3] < threshold:
            a_m.append([i1, i2])

    matches = np.asarray(a_m)
    ### END YOUR CODE

    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

   
   padded1 = pad() Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)

    Return:
        H: a matrix of shape (P, P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    ### YOUR CODE HERE
    H = np.linalg.lstsq(p2, p1)[0]
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = keypoints1[matches[:,0]]
    matched2 = keypoints2[matches[:,1]]
    padded1 = pad(matched1)
    padded2 = pad(matched2)

    #max_inliers = np.zeros(N)
    max_inliers = []
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    for i in range(n_iters):
        m = [p for p in range(N)]
        np.random.shuffle(m)
        m = m[:n_samples]
        H = fit_affine_matrix(matched1[m], matched2[m])
        ins = []
        for j in range(N):
            trans = padded2[j] @ H
            trans = np.array([trans[0]/trans[2], trans[1]/trans[2]])
            di = np.linalg.norm(trans - matched1[j])
            if di < threshold:
                ins.append(j)

        if len(ins) > n_inliers:
            n_inliers = len(ins)
            max_inliers = ins

    H = fit_affine_matrix(matched1[max_inliers], matched2[max_inliers])
    ### END YOUR CODE
    return H, orig_matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. Compute the gradient image in x and y directions (already done for you)
    2. Compute gradient histograms for each cell
    3. Flatten block of histograms into a 1D feature vector
        Here, we treat the entire patch of histograms as our block
    4. Normalize flattened block
        Normalization makes the descriptor more robust to lighting variations

    Args:
        patch: grayscale image patch of shape (H, W)
        pixels_per_cell: size of a cell with shape (M, N)

    Returns:
        block: 1D patch descriptor array of shape ((H*W*n_bins)/(M*N))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    # For each cell, keep track of gradient histrogram of size n_bins
    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    for r in range(rows):
        for c in range(cols):
            for m in range(G_cells.shape[2]):
                for n in range(G_cells.shape[3]):
                    idx = int(theta_cells[r, c, m, n] // degrees_per_bin)
                    if idx == 9:
                        idx = 8
                    cells[r, c, idx] += G_cells[r, c, m, n]
    block = cells.flatten()
    block = (block - block.mean()) / block.std()
    ### YOUR CODE HERE

    return block


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image

    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]

    ### YOUR CODE HERE
    img1_mask = img1_mask.astype(np.float32)
    img2_mask = img2_mask.astype(np.float32)
    decreasing = np.tile(np.linspace(1, 0, right_margin - left_margin), (out_H, 1))
    img1_mask[:, left_margin:right_margin] = decreasing
    increasing = np.tile(np.linspace(0, 1, right_margin - left_margin), (out_H, 1))
    img2_mask[:, left_margin:right_margin] = increasing
    part_left = img1_warped * img1_mask
    part_right = img2_warped * img2_mask
    merged = np.zeros((out_H, out_W), dtype=np.float32)
    merged[:, :left_margin] = part_left[:, :left_margin]
    merged[:, left_margin:right_margin] = part_left[:, left_margin:right_margin] + part_right[:, left_margin:right_margin]
    merged[:, right_margin:] = part_right[:, right_margin:]
    ### END YOUR CODE

    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    # Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)

    return keypoints, descriptors, matches

def stich(imgs, keypoints, descriptors, matches):
    ### YOUR CODE HERE
    transforms = []
    for i in range(len(imgs)-1):
        H, _ = ransac(keypoints[i], keypoints[i+1], matches[i], threshold=1)
        transforms.append(H)
    mid = len(keypoints) // 2
    transforms.insert(mid, np.eye(3))
    Hs = [np.eye(3)] * (len(imgs))

    accu = np.eye(3)
    for i in range(mid + 1, len(imgs)):
        accu = accu @ transforms[i]
        Hs[i] = accu

    accu = np.eye(3)
    for i in range(mid, -1, -1):
        accu = accu @ np.linalg.inv(transforms[i])
        Hs[i] = accu


    output_shape, offset = get_output_space(imgs[mid], imgs, Hs)
    merged = np.zeros(output_shape, dtype=np.float32)
    overlap = np.zeros(output_shape, dtype=np.float32)
    for i in range(len(imgs)):
        warped = warp_image(imgs[i], Hs[i], output_shape, offset)
        m = (warped != -1) # Mask == 1 inside the image
        warped[~m] = 0     # Return background values to 0
        merged += warped
        overlap += m * 1.0
    panorama = merged / np.maximum(overlap, 1)
    ### END YOUR CODE

    return panorama
