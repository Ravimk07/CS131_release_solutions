"""
CS131 - Computer Vision: Foundations and Applications
Assignment 5
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/25/2018
Python Version: 3.5+
"""

import numpy as np
import random
import sys
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        new_assignments = np.zeros(N, dtype=np.uint32)
        for i in range(N):
        	feat = features[i]
        	min_idx = 0
        	min_dist = np.linalg.norm(feat - centers[0], 2)
        	for j in range(1, len(centers)):
        		dist = np.linalg.norm(feat - centers[j], 2)
        		if dist < min_dist:
        			min_dist = dist
        			min_idx = j
        	new_assignments[i] = min_idx
        if np.array_equal(new_assignments, assignments):
        	break

        assignments = new_assignments
        for i in range(len(centers)):
        	assignments_to_center = features[assignments == i]
        	centers[i] = np.sum(assignments_to_center, 0) / assignments_to_center.shape[0]
        ### END YOUR CODE

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        new_assignments = np.zeros(N, dtype=np.uint32)
        expanded_features = np.repeat(np.expand_dims(features, 0), k, axis=0)
        expanded_centers = np.expand_dims(centers, 1)
        dists = np.linalg.norm(expanded_features - expanded_centers, 2, axis=2)
        new_assignments = np.argmin(dists, axis=0)

        if np.array_equal(new_assignments, assignments):
        	break

        assignments = new_assignments
        for i in range(len(centers)):
        	assignments_to_center = features[assignments == i]
        	centers[i] = np.sum(assignments_to_center, 0) / assignments_to_center.shape[0]
        ### END YOUR CODE

    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N

    dict_centers = {i: {'center':features[i], 'indices':[i]} for i in range(centers.shape[0])}
    dict_distances = {}
    next_center_key = centers.shape[0]
    for i in range(centers.shape[0]):
        for j in range(i + 1, centers.shape[0]):
            dict_distances[(i, j)] = np.linalg.norm(dict_centers[i]['center'] - dict_centers[j]['center'], 2)
    
    k_cls = k
    while n_clusters > k_cls:
        ### YOUR CODE HERE
        min_dist = sys.float_info.max
        min_key = (-1, -1)
        for k in dict_distances.keys():
            d = dict_distances[k]
            if d < min_dist:
                min_dist = d
                min_key = k
        new_center = (dict_centers[min_key[0]]['center'] + dict_centers[min_key[1]]['center']) / 2
        dict_centers[next_center_key] = {
                'center': new_center, 
                'indices': dict_centers[min_key[0]]['indices'] + dict_centers[min_key[1]]['indices']}
        del dict_centers[min_key[0]]
        del dict_centers[min_key[1]]

        for k in list(dict_distances.keys()):
            if min_key[0] in k or min_key[1] in k:
                del dict_distances[k]

        for k in dict_centers:
            v = dict_centers[k]
            if k == next_center_key:
                continue
            dict_distances[(k, next_center_key)] = np.linalg.norm(v['center'] - new_center, 2)

        next_center_key += 1
        n_clusters = len(list(dict_centers.keys()))

    for i, k in enumerate(dict_centers):
        assignments[dict_centers[k]['indices']] = i

    ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    features = img.reshape(H*W, C)
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    features[:, :3] = img.reshape(H*W, C)
    grid = np.mgrid[:H, :W]
    features[:, 3] = grid[0].reshape(H*W)
    features[:, 4] = grid[1].reshape(H*W)

    features = (features - np.mean(features, axis=0)) / np.std(features,  axis=0)
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    accuracy = np.sum(mask_gt == mask) / (mask.shape[0] * mask.shape[1])
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
