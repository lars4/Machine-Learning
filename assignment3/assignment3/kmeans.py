import numpy as np
import time


def kmeans(X, k, max_iter=100):
    """
    Perform k-means clusering on the data X with k number of clusters.

    Args:
        X: The data to be clustered of shape [n, num_features]
        k: The number of cluster centers to be used

    Returns:
        centers: A matrix of the computed cluster centers of shape [k, num_features]
        assign: A vector of cluster assignments for each example in X of shape [n] 

    """

    centers = None
    assign = None
    i=0
    
    start = time.time()


    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of the input data X and store the         #
    # resulting cluster-centers as well as cluster assignments.           #
    #                                                                     #
    #######################################################################
    
    # 1st step: Chose k random points of X as initial cluster centroids.
    # Centroids: Data points that are randomly plotted on the graph,
    #            eventually they're going to be the centers of our
    #            k clusters. [centers]

    # How many clusters do we want to find?
    # X.shape[0] is our number of instances in X.
    # X.shape[1] would be the number of features.
    # 'np.zeros(X.shape[0]' assigns an initial data point, which
    # is a vector (basically an array) filled with 'n' many zeros.
    centers = X[np.random.randint(0, X.shape[0] - 1, k)]
    assign = np.zeros(X.shape[0])
    
    # Define distance measurement method.
    # dist_method = euclidian
    # Just learned about NumPy's 'linalg.norm()'...

    for z in range(max_iter):
        prev_assign = np.array(assign)

        # 2nd step: Update the cluster assignment.
    
        for i in range(X.shape[0]):
            # 'x' equals instace at position 'i' of our dataset X.
            x = X[i]
            # Calculate euclidian distance between data point 'x' and our k centroids.
            # 'np.tile(A, reps)' adjusts dimension, so 'x' and 'centers' are operationable against each other.
            diff = np.linalg.norm((np.tile(x, (k, 1)) - centers), None, axis = 1)
            # Assigns closest centroid to data point.
            assign[i] = np.argmin(diff)

        # 3rd step: Check for convergence.
        # If none of the cluster assignments change, we're good.
        # Otherwise 'max_iter' is our threshold.

        if np.array_equal(prev_assign, assign):
            break

        # 4th step: Update the cluster centers based on the new assignment.

        for j in range(centers.shape[0]):
            numinator = np.where(assign == j, np.ones(assign.shape[0]), np.zeros(assign.shape[0]))
            denumiator = np.sum(numinator)
            numinator = np.dot(numinator, X)
            centers[j] = numinator / denumiator

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    exec_time = time.time()-start
    print('Number of iterations: {}, Execution time: {}s'.format(i+1, exec_time))
    print('Number of iterations of the k-mean function: {}'.format(z))
    
    return centers, assign