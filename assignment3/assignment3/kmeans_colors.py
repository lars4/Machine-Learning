from sklearn.cluster import KMeans
import numpy as np
import sklearn


def kmeans_colors(img, k, max_iter=100):
    """
    Performs k-means clustering on the pixel values of an image.
    Used for color-quantization/compression.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of color clusters to be computed

    Returns:
        img_cl:  The color quantized image of shape [h, w, 3]

    """

    img_cl = None

    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of on the pixel values of the image img.  #
    #######################################################################
    
    # Convert to floats instead of the default 8 bits integer coding.
    img = np.array(img, dtype = np.float64) / 255
    w, h, d = original_shape = tuple(img.shape)
    
    # Transforming image to a 2D numpy array.
    img_array = np.reshape(img, (w * h, d))
    
    # Using 2000 points to train our K-means.
    img_array_sample = sklearn.utils.shuffle(img_array, random_state = 0)[:2000]
    
    # Using K-means to create k(4) clusters (which are distinct colors).
    kmeans = KMeans(n_clusters = k, random_state = 0, max_iter = max_iter)
    
    # Fit array sample on K-means with above specs.
    kmsample = kmeans.fit(img_array_sample)
    
    # Get labels for all points.
    labels = kmeans.predict(img_array)
    
    # Recreate image.
    img_cl = np.zeros((w, h, d))
    label_idx = 0
    
    # For each pixel we set the color of its nearest centroid.
    for i in range(w):
        for j in range(h):
            img_cl[i][j] = kmeans.cluster_centers_[labels[label_idx]]
            label_idx += 1
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return img_cl