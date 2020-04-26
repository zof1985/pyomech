


########    IMPORTS    ########



import numpy as np



########    METHODS    ########



def crosstab(X):
    """
    provide the contingency table of the entered arrays according to the entering order.
    
    Input:
        X:      (nD array)
                the multidimensional array along which the contingency table must be generated

    Output:
        C:      (ndarray)
                an n-dimensional matrix with 1 dimension per entered variable defining the number of occurrences of
                each combination

        P:      (list)
                the list of indices corresponding to the values in each dimension
    """

    # get the unique combinations of columns in M with counts
    U, F = np.unique(X, axis=0, return_counts=True)

    # get the unique elements in each array
    Z = [np.unique(i) for i in X.T]

    # get an empty grid of zeros defined by the unique elements in each provided array
    C = np.zeros([len(i) for i in Z], dtype=int)

    # get the C coordinates corresponding to each unique combination in U and add the coorect frequencies
    for i in np.arange(len(F)):
        C[tuple([np.argwhere(U[i][j] == Z[j]).flatten()[0] for j in np.arange(len(Z))])] = F[i]

    # return the table
    return C, Z



def entropy(X, Y=None, base=2):
    """
    Calculate the joint entropy of X or the conditional entropy of X given Y.
    
    Input:
        X:              (2D array)
                        Each row will be an observation, while each column a variable. Then, the joint entropy of X
                        is calculated as:
                                                        H(X) = sum(p(X) log(p(X)))
                        Where X is a n-dimensional vector.
        
        Y:              (2D array or None)
                        Each row will be an observation, while each column a variable. Then, the joint entropy of Y
                        is calculated as:
                                                        H(Y) = sum(p(Y) log(p(Y)))
                        Where Y is a n-dimensional vector.
                        Then, the conditional entropy of X given Y is obtained via:
                                                        H(X|Y) = H(X, Y) - H(Y) 

        base:           (int)
                        the base of the logarithm to be used.

    Output:
        H:              (float)
                        the Joint or Conditional Entropy.
    """

    # if Y is provided, recursively calculate H(X|Y) = H(X, Y) - H(Y)
    if Y is not None:
        Hy = entropy(Y, base=base)
        Hxy = entropy(np.concatenate(X, Y), base=base)
        return Hxy - Hy
    
    # get the contingency table of X
    X_c = crosstab(X)

    # normalize the contingency table in order to obtain probabilities
    X_p = X_c / np.sum(X_c)

    # get the entropy
    return -np.sum(X_p * (np.log(X_p) / np.log(base)))
    


def mutual_information(X=None, Y=None, base=2):
    """
    Calculate the mutual information of the variables in X or the conditional mutual information of the variables in
    X given the variables in Y.
    
    Input:
        X:              (2D array)
                        Each row will be an observation, while each column a variable.
        
        Y:              (2D array)
                        If not None, each row will be an observation, while each column a variable. Then conditional
                        mutual information is returned.

        base:           (int)
                        the base of the logarithm to be used.

    Output:
        M:              (float)
                        The joint or conditional mutual information.
    """

    # get the order of X
    N = X.shape[1]

    # if N < 2 no mutual information can be calculated
    if N < 2:
        raise AttributeError("X must have a minimum of 2 columns.")
    
    # calculate the mutual information as sum of entropies
    return -np.sum([(-1) ** (i + 1) * entropy(np.atleast_2d(X[:, i].flatten()).T, Y, base) for i in np.arange(N)])



def total_correlation(X=None, Y=None, base=2):
    """
    Calculate the total correlation of the variables in X or the total conditional correlation of the variables in X
    given the variables in Y.
    
    Input:
        X:              (2D array)
                        Each row will be an observation, while each column a variable.
        
        Y:              (2D array)
                        If not None, each row will be an observation, while each column a variable. Then conditional
                        mutual information is returned.

        base:           (int)
                        the base of the logarithm to be used.

    Output:
        C:              (float)
                        The joint or conditional mutual information.
    """

    # get the order of X
    N = X.shape[1]
            
    # calculate the total correlation as the sum of the joint or conditional entropies resulting from single variables
    # minus the joint or conditional entropy of X.
    return np.sum([entropy(np.atleast_2d(X[:, i].flatten()).T, Y, base) for i in np.arange(N)]) - entropy(X, Y, base)