


########    IMPORTS    ########



import numpy as np



########    METHODS    ########



def crosstab(*args, **kwargs):
    """
    provide the contingency table of the entered arrays according to the entering order.
    
    Input:
        args/kwargs:    (set of 1D arrays)
                        the arrays along which the contingency table must be generated

    Output:             (ndarray)
                        an n-dimensional matrix with 1 dimension per entered variable defining the number of
                        occurrences of each combination
    """

    # put all the variables into a 2D array where each row contains the samples and each column is one of the entered
    # arrays
    M = np.vstack(np.atleast_2d([*args, *[kwargs[i] for i in kwargs]]))

    # get the unique combinations of columns in M with counts
    U, F = np.unique(M, axis=1, return_counts=True)

    # get the unique elements in each array
    Z = [np.unique(i) for i in M]

    # get an empty grid of zeros defined by the unique elements in each provided array
    C = np.zeros([len(i) for i in Z])

    # for each unique combination, get the counts in the right place
    for i in np.arange(len(F)):

        # get the coordinates
        coords = [np.argwhere(U[i][j] == Z[j]).flatten()[0] for j in np.arange(len(Z))]

        # store the counts in the right place
        C[*coords] = F[i]

    # return the table
    return C



def entropy(x=None, y=None, con_tab=None, base=2, conditional=True):
    """
    Calculate the joint or conditional entropy between x and y according to the rule:

        Joint:
                    H(X;Y) = - sum ( p(x, y) log p(x, y) )  |
                              x in X
                              y in Y

        Conditional:
                                       /             p(x, y) \
                    H(X|Y) = -  sum   | p(x, y) log --------  |
                               x in X  \              p(y)   /
                               y in Y
    Where:
        p(x, y)     =   joint probability of the x, y combination
           p(y)     =   conditional probability of y

    Input:
        x:              (1D array)
                        the data in x
        
        y:              (1D array)
                        the data in y
        
        con_tab:        (2D array)
                        the contingency table containing the counts of x and y. If provided, con_tab must show the
                        counts of each value of x as rows and each value of y inx and y are ignored. 

        base:           (int)
                        the base of the logarithm to be used.

        conditional:    (bool)
                        if True, the conditional entropy is returned. Otherwise the Joint Entropy is provided
    """
    
    # get the contingency table
    if con_tab is None:
        con_tab = crosstab(x, y).values
    
    # normalize the contingency table in order to obtain probabilities
    p_tab = con_tab / np.sum(con_tab)

    # get the p_y and replicate it for the number of elements in x
    if conditional:
        p_y = np.sum(p_tab, axis=0).flatten()
        p_y = np.atleast_2d([p_y for i in np.arange(p_tab.shape[0])])
    else:
        p_y = 1

    # get the entropy
    return - np.sum(p_tab * (np.log(p_tab / p_y) / np.log(base)))
    


def mutual_information(x=None, y=None, z=None, con_tab=None, base=2):
    """
    Calculate either the Mutual Information score I(X;Y) or the Conditional Mutual Information I(X;Y|Z) according to
    the rule:
                                                                 /              p(x, y)   \
                    I(X;Y) = H(X,Y) - H(X|Y) - H(Y|X) = - sum   | p(x, y) log -----------  |
                                                         x in X  \            p(y) * p(x) /
                                                         y in Y
                    
                                        /            p(x, y, z) * p(z) \
                    I(X;Y|Z) = - sum   | p(x, y) log -----------------  |
                                x in X  \             p(y|z) * p(x|z)  /
                                y in Y
                                z in Z
    Where:
        H(X,Y) = Joint Entropy of X and Y
        H(X|Y) = Conditional entropy of X given Y
        H(Y|X) = Conditional entropy of Y given X
    
    Input:
        x:              (1D array)
                        the data in x
        
        y:              (1D array)
                        the data in y
        
        z:              (1D array)
                        the data of a third array. If provided, Conditional Mutual information is provided.
        
        con_tab:        (2D array)
                        the contingency table containing the counts of x and y. If provided, con_tab must show the
                        counts of each value of x as rows and each value of y inx and y are ignored. 

        base:           (int)
                        the base of the logarithm to be used for calculating the.
    """



