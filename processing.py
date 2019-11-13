# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:12:24 2018

@author: lzoffoli
"""



def psd(y, fs=1, n=None):
    """
    compute the power spectrum of y using fft

    Input:
        y: (ndarray)
            A 1D numpy array

        fs: (float)
            the sampling frequency

        n: (None, int)
            the number of samples to be used for FFT. if None, the length of y is used.

    Output:
        P: (ndarray)
            the power of each frequency

        F: (ndarray)
            the frequencies.
    """

    # dependancies
    from pyomech.utils import classcheck
    import numpy as np

    # check the data
    classcheck(y, ["ndarray"])
    assert y.ndim == 1, "'y' must be a 'ndarray' of dimension 1."
    classcheck(fs, ["float"])
    assert fs > 0, "'fs' must be > 0."
    classcheck(n, ["NoneType", "int"])

    # set n
    if n is None: n = len(y)

    # get the FFT and normalize by the length of y
    y = np.fft.rfft(y, n) / len(y)

    # get the amplitude of the signal
    a = abs(y)

    # get the power of the signal
    p = np.concatenate([[a[0]], 2 * a[1:-1], [a[-1]]]).flatten() ** 2

    # get the frequencies
    f = np.linspace(0, fs / 2, len(p))

    # return the data
    return p, f



def nextpow(n, b):
    """
    calculate the power at which the base has to be elevated in order to get the next value to n.

    Input:
        n: (float)
            a number
        b: (float)
            the base of the power.

    Output:
        k: (float)
            the value verifying the equation: n = b ** k --> k = log(n) / log(b)
    """

    # import dependancies
    from numpy import log

    # check dependancies
    assert n.__class__.__name__[:5] == "float" or n.__class__.__name__[:3] == "int", "'n' must be a float."
    assert b.__class__.__name__[:5] == "float" or b.__class__.__name__[:3] == "int", "'b' must be a float."

    # return k
    return log(n) / log(b)



def power_freq(X, C, fs=1):
    """
    get the cumulative signal power corresponding to C.

    Input:
        X: (ndarray)
            The signal having 1 dimension.
        C: (float)
            the power percentage within the (0, 100] range
        fs: (float)
            the sampling frequency in Hz.

    Output:
        - the frequency (in Hz) corresponding to a cumulative signal power equal to C.
        - the frequencies tested.
        - the cumulative power.

    References:
        Sinclair J, Taylor PJ, Hobbs SJ.
            Digital filtering of three-dimensional lower extremity kinematics: an assessment.
            J Hum Kinet. 2013;39(1):25–36.
    """

    # import the necessary packages
    import numpy as np

    # check the entered data
    assert X.__class__.__name__ == "ndarray", "'X' must be an ndarray with dimension 1."
    assert X.ndim == 1, "'X' must be an ndarray with dimension 1."
    assert C > 0 and C <= 100, '"C" must be in the (0, 100] range.'

    # get the power spectral density of the signal
    X = X.flatten() - np.mean(X)
    P, F = psd(X, fs)
    if np.isclose(np.sum(P), 0): return 0, F, P
    P = 100 * np.cumsum(P) / np.sum(P)
    if np.all(X == 0): return F[-2], F, P
    try: return F[P >= C][0], F, P
    except Exception: return F[-2], F, P



def res_an(X, fs=1, K=100, N=4, S=2, P=5, par=None):
    """
    Perform residual analysis of the entered data according to the Residual Analysis procedure of Winter (2009).

    Input:
        X: (nd array)
            a 1D array with the signal to be investigated.
        fs: (float)
            the sampling frequency of the signal.
        K: (int)
            the number of frequencies tested within the (0, fs/2) range.
        N: (int)
            the other of the Butterworth phase-corrected filter.
        S: (int)
            the number of segments to be used to find the best deflection point.
        P: (int)
            the minimum number of elements being part of each segment.
        par: (None or Pool)
            parallel object.

    Output:
        - the optimal cutoff frequency (in Hz) according to references.
        - the tested frequencies.
        - the residuals.
        - the crossovers.

    References:
        Winter DA.
            Biomechanics and Motor Control of Human Movement. Fourth Edi.
            Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
        Lerman PM.
            Fitting Segmented Regression Models by Grid Search. Appl Stat. 1980;29(1):77.
    """

    # import the required packages
    import numpy as np

    # check the entered data
    assert X.__class__.__name__ == "ndarray", "'X' must be an ndarray with 1 dimension."
    assert X.ndim == 1, "'X' must be an ndarray with 1 dimension."
    assert fs.__class__.__name__ in ["float", "int"], '"fs" must be a float or int number.'
    assert K.__class__.__name__ == "int", '"K" must be a float or int number.'
    assert N.__class__.__name__ == "int", '"N" must be a float or int number.'
    assert S.__class__.__name__ == "int", '"S" must be a float or int number.'
    assert P.__class__.__name__ == "int", '"P" must be a float or int number.'
    assert par.__class__.__name__ in ["NoneType", "Pool"], "'par' must be None or a Pool object."

    # get the frequency span
    freqs = np.linspace(0, fs * 0.5, K + 2)[1:-1]

    # if X is constant return 0
    if np.float32(np.nansum(X - np.nanmean(X))) == 0:
        return freqs[0], freqs, np.tile(freqs[0], len(freqs)), np.arange(S) + P

    # get the residuals
    Q = [SSE(X, filt(X, N, i, fs)) if par is None else par.apply(SSE, args=(X, filt(X, N, i, fs))) for i in freqs]
    Q = np.array(Q).flatten()

    # check if the signal returned constant residuals
    if np.all(np.diff(Q) == 0):
        opt = 0
        Z = np.array([0])
    else:

        # define the residuals method to be used for the calculation of the crossovers
        def res(x, y, s):
            C = [np.arange(s[i], s[i + 1] + 1) for i in np.arange(len(s) - 1)]  # get the coordinates
            Z = [np.polyfit(x[i], y[i], 1) for i in C]  # get the fitting parameters for each interval
            V = [np.polyval(v, x[C[i]]) for i, v in enumerate(Z)]  # get the regression lines for each interval
            return np.sum([SSE(y[C[i]], v) for i, v in enumerate(V)])  # get the sum of squared residuals

        # get the optimal crossingover point that separates the S regression lines best fitting the residuals data.
        Z = crossovers(Q, None, S, P, res, False, par)[0]

        # get the intercept of the second line (i.e. the most flat one)
        T = np.polyfit(freqs[Z[-1]:], Q[Z[-1]:], 1)[-1]

        # get the optimal cutoff
        opt = freqs[np.argmin(abs(Q - T))]

    # get the cutoff frequency
    return opt, freqs, Q, Z



def SSE(x, y):
    """
    Return the sum of squared residuals between x and y

    Input:
        x: (ndarray)
            one signal
        y: (ndarray)
            another signal

    Output:
        sum(abs(x - y) ** 2)
    """

    # check input data
    assert x.__class__.__name__ == "ndarray", "'x' must be a ndarray."
    assert y.__class__.__name__ == "ndarray", "'y' must be a ndarray."

    # import dependancies
    from numpy import sum as nsum

    # return data
    return nsum((x - y) ** 2)



def crossovers(Y, X=None, K=2, samples=5, res=None, plot=False, par=None):
    """
    Detect the position of the crossing over points between K regression lines used to best fit the data.

    Procedure:
        1)  Get all the segments combinations made possible by the given number of crossover points.
        2)  For each combination, calculate the regression lines corresponding to each segment.
        3)  For each segment calculate the residuals between the calculated regression line and the effective data.
        5)  Once the sum of the residuals have been calculated for each combination, sort them by residuals amplitude.

    References
        Lerman PM.
            Fitting Segmented Regression Models by Grid Search. Appl Stat. 1980;29(1):77.

    Input:
        Y: (ndarray)
            The data to be fitted.
        X: (ndarray or None)
            The x-axis data. If not provided it will be set as range(N) with N equal to the length of Y.
        K: (int)
            The number of regression lines to be used to best fit Y.
        samples: (int)
            The minimum number of points to be used to calculate the regression lines.
        res: (None or method)
            A method accepting X, Y and S where S is a list of crossover points splitting X and Y. It must return a
            float value representing the residuals of the fitting of the current crossovers layout. If None, the default
            method returns the sum of the squared residuals as measure of error.
        plot: (bool)
            If True a plot showing the crossing over points and the calculated regression lines is generated.
        par: (None or Pool)
            parallel object.

    Output:
        An ordered array of indices where the columns reflect the position (in sample points) of the crossing overs
        while the rows shows the best fitting options from the best to the worst.
    """

    # import the required packages
    import numpy as np
    import matplotlib.pyplot as pl
    import pyomech.plot as pp
    from os import remove
    from itertools import product

    # check the entered data
    assert Y.__class__.__name__ == "ndarray", "'Y' must be a ndarray object. with 1 dimension."
    assert Y.ndim == 1, "'Y' must be a ndarray object. with 1 dimension."
    if X is None: X = np.arange(len(Y))
    assert X.__class__.__name__ == "ndarray", "'X' must be a ndarray object. with 1 dimension."
    assert X.ndim == 1, "'X' must be a ndarray object. with 1 dimension."
    assert len(X) == len(Y), "X must have the same length as Y."
    assert samples.__class__.__name__ == "int", "'samples' must be an int."
    assert samples >= 2, "'samples' must be an integer >= 2."
    assert K.__class__.__name__ == "int", "'K' must be an int."
    assert K >= 1, "'K' must be an integer >= 1."
    assert plot.__class__.__name__ == "bool", "'plot' must be a bool object."
    assert par.__class__.__name__ in ["NoneType", "Pool"], "'par' must be None or a Pool object."
    if res is None:
        def resid(x, y, s):
            C = [np.arange(s[i], s[i + 1] + 1) for i in np.arange(len(s) - 1)]  # get the coordinates
            Z = [np.polyfit(x[i], y[i], 1) for i in C]  # get the fitting parameters for each interval
            V = [np.polyval(v, x[C[i]]) for i, v in enumerate(Z)]  # get the regression lines for each interval
            return np.sum([np.sum((y[C[i]] - v) ** 2) for i, v in enumerate(V)])  # get the sum of squared residuals
    else:
        resid = res

    # get all the possible combinations
    J = [j for j in product(*[np.arange(samples * i, len(Y) - samples * (K - i)) for i in np.arange(1, K)])]

    # remove those combinations having difference between intervals below S
    J = [i for i in J if np.all(np.diff(i) >= samples)]

    # generate the crossovers matrix
    J = np.hstack((np.zeros((len(J), 1)), np.atleast_2d(J), np.ones((len(J), 1)) * len(Y) - 1)).astype(int)

    # calculate the residuals for each combination
    R = np.array([resid(X, Y, i) if par is None else par.apply(resid, args=(X, Y, i)) for i in J])

    # sort the residuals
    T = np.argsort(R)

    # plot the output if required
    if plot:
        x = J[T[0]]
        x = [np.arange(x[i], x[i + 1] + 1) for i in np.arange(len(x) - 1)]
        z = [np.polyfit(i, Y[i], 1) for i in x]
        z = [np.polyval(z[i], x[i]) for i in np.arange(len(x))]
        fig, ax = pl.subplots(1, 1, figsize=(pp.cm2in(20), pp.cm2in(20)), dpi=1000, frameon=False)
        pl.plot(X, Y, color="navy", alpha=0.5)
        for i in np.arange(len(x)): pl.scatter(x[i], z[i], color='darkred')
        pp.set_layout(ax, "X", "Y")

    # get the optimal crossovers order
    return X[J[T, 1:-1]]



def filt(X, N, C, F=None, T='lowpass', P=True):
    """
    Provides a convenient function to call a phase-corrected, low-pass, Butterworth filter with the specified parameters.

    Input:
        X: (1D numpy array)
            the signal to be filtered.
        N: (int > 0)
            the order of the filter.
        C: (float > 0)
            if F is provided, C is the cutoff of the filter in Hz.
            Otherwise, C will be the normalized frequency (i.e. freq / (0.5 * sampling freq))
        F: (float > 0)
            the sampling frequency of the signal in Hz.
        T: (str)
            a string defining the type of the filter: e.g. "low", "high", "bandpass", etc.
        P: (bool)
            should the filter be applied twice in opposite directions to correct for phase lag?

    Output:
        the resulting 1D filtered signal
    """

    # import the required packages
    from scipy.signal import butter, filtfilt, lfilter
    from numpy import array

    # check X
    assert X.__class__.__name__ == "ndarray", "'X' must be a 'ndarray' with ndim = 1."
    assert X.ndim == 1, "'X' must be a 'ndarray' with ndim = 1."

    # check N
    assert N.__class__.__name__ == "int", "'N' must be an 'int'."
    assert N > 0, "'N' must be > 0."

    # check F
    F = float(F)
    assert F > 0, "'F' must be > 0."

    # check T
    valid_filters = ['lowpass', 'highpass', 'bandpass', 'bandstop', 'low', 'high']
    assert T.lower() in valid_filters, "'T' must be any of " + str(valid_filters)

    # check C
    C = array([C]).flatten()
    assert all([i > 0 for i in C]), "All values in C must be > 0."
    assert all([i < F * 0.5 for i in C]), "All values in C must be < F/2."
    if T in ['bandpass', 'bandstop']: assert len(C) == 2, "'C' must have 2 cutoff values."
    else: assert len(C) == 1, "'C' must have 1 cutoff."

    # check P
    assert P or not P, "'P' must be True or False."

    # get the filter coefficients
    (B, A) = butter(N, (C / (1 if F is None else (0.5 * F))), T)

    # return the filtered data
    return filtfilt(B, A, X.flatten()) if P else lfilter(B, A, X.flatten())



def interpolate(Y, N=101, X_old=None, X_new=None):
    """
    Get the cubic spline interpolation of Y to N points

    Input:
        Y: (1D array)
            the data to be interpolated.
        N: (int)
            the number of points for the interpolation. It is ignored if X_new is provided.
        X_old: (1D array)
            the X axis data corresponding to Y. If not provided, it is set as an increasing set of int with length
            equal to Y.
        X_new: (1D array)
            the X axis that Y needs to correspond to. If provided, N is ignored and Y is fitted to X_new.

    Output:
        the Y axis cubic spline interpolated to N points or X_new.
    """

    # import the required packages
    from scipy.interpolate import splev, splrep
    import numpy as np

    # check Y
    assert Y.__class__.__name__ == "ndarray", "'Y' must be a 'ndarray' with ndim = 1."
    assert Y.ndim == 1, "'Y' must be a 'ndarray' with ndim = 1."

    # check X_old
    if X_old is None: X_old = np.arange(len(Y))
    assert X_old.__class__.__name__ == "ndarray", "'X_old' must be a 'ndarray' with ndim = 1."
    assert X_old.ndim == 1, "'X_old' must be a 'ndarray' with ndim = 1."

    # ensure X_old is monotonic
    idx = np.unique(X_old, return_index=True)[1]
    x = X_old[idx]
    y = Y[idx]
    assert np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0), "'X_old' must be monotonic."

    # check N
    assert N.__class__.__name__ in ["int", "NoneType"], "'N' must be an int > 0 or None."
    if N is not None: assert N > 0, "'N' must be an int > 0."
    assert (N is not None) or (X_new is not None), "'N' and 'X_new' cannot be both None."

    # check X_new
    if X_new is None:
        sign = np.unique(np.sign(np.diff(x)))
        init = np.min(X_old) if sign > 0 else np.max(X_old)
        stop = np.max(X_old) if sign > 0 else np.min(X_old)
        X_new = np.linspace(init, stop, N)
    assert X_new.__class__.__name__ == "ndarray", "'X_new' must be a 'ndarray' with ndim = 1."
    assert X_new.ndim == 1, "'X_new' must be a 'ndarray' with ndim = 1."

    # generate the interpolated axis
    return splev(X_new, splrep(x, y, k=3, s=0)), X_new



def find_crossings(x, value=0.):
    """
    Dectect the crossing points in x compared to value.

    Input:
        x: (1D array)
            the data.
        value: (float value or a 1D array with the same shape of x)
            the value/s to be used to detect the crossings. If value is an array, the function will find those points
            in x crossing values according to the value of values at a give location.

    Output:
        a numpy array with the location of the crossing points

    """

    # import the required packages
    from numpy import array, tile, argwhere, sign, diff

    # check data
    assert x.__class__.__name__ == "ndarray", "'x' must be an ndarray object."
    assert x.ndim == 1, "Only 1D arrays can be processed."

    # cast the input to be a 1-d array
    V = array([value]).flatten()
    V = tile(V, len(x)) if len(V) == 1 else V
    assert len(V) == len(x), "'value' must be a scalar or an array with length equal to x."

    # look for the crossing points
    return argwhere(abs(diff(sign(x - V))) == 2).flatten() + 1


# ROTATION MATRIX EDITOR
def angle2rmat(angle, dimensions=3, axis=1):
    """
    Create a rotation matrix along the i-th axis of an N-dimensional vector. The function returns the rotation matrix T
    which satisfies:
                            R_kn = T_nn(a, i) x M_kn
    where M and R are generic k-by-n matrices and T_nn is the rotation matrix that rotates M_kn along the i-th
    dimension of "a" radiants.

    :param angle: the angle of rotation. it can be a simpy symbol or a float number.
    :param dimensions: the number of dimensions of the rotation matrix.
    :param axis: the axis along which the rotation have to occur.
    :return: the sympy Matrix or numpy ndarray resulting in the desired rotation matrix.
    """

    # import the required packages
    import numpy as np
    import sympy as sy

    def getsym(i, j, axis, angle):
        """
        Generate an internal function which will provide the correct sympy symbol for the current coordinate in the
        resulting rotation matrix.

        NOTE: Since the function is internal to rmat, it is assumed that the entered parameters are correct. Thus,
        no check on the entered parameters is performed.

        :param i: the actual row in the rotation matrix.
        :param j: the actual column in the rotation matrix.
        :param axis: the axis along which the rotation occurs.
        :param angle: the sympy symbol representing the angle.
        :return: the symbol corresponding to the current rotation matrix coordinate.
        """

        # the current position is on the main diagonal, thus return 1 if the coordinate is the one of the rotation
        # axis. Otherwise return cos(s).
        if i == j:
            return sy.Float(1) if i == j == axis else sy.cos(angle)

        # the elements outside the main diagonal of the rotation matrix will be sin(s) with sign corresponding to
        # (-1) ** (i + j) for the elements above the diagonal, and -((-1) ** (i + j)) for those below it.
        else:
            if np.any(np.array([i, j]) == axis):
                return sy.Float(0)
            else:
                return sy.Float(-1 if i > j else 1) * sy.Float(np.sign((-1) ** (i + j))) * sy.sin(angle)

    # check the entered data
    if not angle.__class__.__name__ == "Symbol":
        assert angle.__class__.__name__ == 'float', "'angle' must be str or float."
    assert dimensions.__class__.__name__ == 'int', "'dimensions' must be an int."
    assert dimensions > 1, "'dimensions' must be higher than 1."
    assert axis.__class__.__name__ == 'int', "'axis' must be an int."
    assert 0 <= axis < dimensions, "'axis' must be a positive int lower than dimensions."

    # generate the symbol corresponding to the angle
    s = sy.Float(angle) if not angle.__class__.__name__ == "Symbol" else angle

    # create the string which will build the rotation matrix
    r = sy.Matrix([[getsym(i, j, axis, s) for j in np.arange(dimensions)] for i in np.arange(dimensions)])

    # convert the matrix to a numpy ndarray if the given angle was numeric
    return r if angle.__class__.__name__ == "Symbol" else np.atleast_2d(r).astype(float)


def eul2rmat(angles, order=[0, 1, 2]):
    """
    Return the rotation matrix defined by the euler angles and the rotation "order" provided.
    :param angles: a list of angles in radiants.
    :param order: a list containing the axis order.
    :return: a 2D numpy array with the rotation matrix.
    """
    
    # import the necessary packages
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pyomech.utils import classcheck

    # check the entered parameters
    if not angles.__class__.__name__ in ['ndarray']:
        angles = np.arrany([angles]).flatten()
    assert angles.ndim == 1, "'angles' must be a 1D array."
    [classcheck(i, ['float', 'int']) for i in angles]
    assert len(angles) <= 3, "'angles' must have length <= 3."
    if not order.__class__.__name__ in ['ndarray']:
        order = np.array([order]).flatten()
    assert order.ndim == 1, "'order' must be a 1D array."
    [classcheck(i, ['int']) for i in order]
    assert np.all(order < 3), "'order' values must be int < 3."
    assert len(order) <= 3, "'order' must have length <= 3."
    assert len(order) == len(angles), "'order' and 'angles' must have equal length."
    
    # define the order
    str_order = {0: 'x', 1: 'y', 2: 'z'}
    str_order = "".join([str_order[i] for i in order])

    # return the rotation matrix
    return Rotation.from_euler(str_order, angles).as_dcm()


def rmat2eul(rmat, order=[0, 1, 2]):
    """
    Return the Euler angles defining the rotation matrix "rmat" according to the rotation axis order "order".
    :param rmat: a rotation matrix in3 dimensions, i.e. a squared matrix with det=1
    :param order: the order of rotation provided as a list of int reflecting the dimension along which the rotation
    is conducted. E.g. order=[2, 1, 0] will calculate the rotation angles assuming that the rotation matrix follows
    the given order.
    :return: a numpy array with the angles (in radiants) sorted according to the entered rotation order. In addition
    the Scipy.spatial.transform.Rotation class object created to perform the calculations is also returned.
    """

    # import the necessary packages
    import numpy as np
    from scipy.spatial.transform import Rotation

    # check the entered parameters
    txt = "'rmat' must be a rotation matrix."
    assert rmat.ndim == 2 and np.sum(np.diff(rmat.shape)) == 0, txt
    assert np.all(np.array([i for i in rmat.shape]) == 3), "'rmat' must be a 3x3 matrix."
    rmat = np.atleast_2d(rmat).astype(float)
    txt = "'order' must be of length equal to the number of 'rmat' dimensions."
    assert np.all(np.array(order).flatten() < rmat.shape[0]), txt
    order = np.array([order]).flatten()
    assert np.sum(np.floor(order) - order) == 0, "'order' elements must be int."
    assert np.all(order >= 0), "'order elements must be >= 0."
    assert len(order) == 3, "length of the order variable must be 3."

    # define the order
    str_order = {0: 'x', 1: 'y', 2: 'z'}
    str_order = "".join([str_order[i] for i in order])

    # define the rotation class object
    R = Rotation.from_dcm(rmat)
    
    # get the Euler angles
    return R.as_euler(str_order), R


def rmat2rvec(rmat):
    """
    Return the rotation vector (axis-angle) defining the rotation matrix "rmat".
    :param rmat: a rotation matrix in3 dimensions, i.e. a squared matrix with det=1
    :return: a numpy array with the rotation vector and the Scipy.spatial.transform.Rotation
    class object created to perform the calculations.
    """

    # import the necessary packages
    import numpy as np
    from scipy.spatial.transform import Rotation

    # check the entered parameters
    txt = "'rmat' must be a rotation matrix."
    assert rmat.ndim == 2 and np.sum(np.diff(rmat.shape)) == 0, txt
    assert np.all(np.array([i for i in rmat.shape]) == 3), "'rmat' must be a 3x3 matrix."
    rmat = np.atleast_2d(rmat).astype(float)

    # define the rotation class object
    R = Rotation.from_dcm(rmat)
    
    # get the Euler angles
    return R.as_rotvec(), R


def rvec2rmat(rvec):
    """
    Return the rotation matrix corresponding to an (axis-angle) rotation vector.
    :param rvec: a 3x1 nd array defining the components of the rotation vector.
    :return: a 3x3 numpy array defining the rotation matrix corresponding to rvec.
    In addition, also the scipy.spatial.transform.Rotation used to perform the calculation
    is returned.
    """

    # import the necessary packages
    import numpy as np
    from scipy.spatial.transform import Rotation

    # check the entered parameters
    rvec = np.array([rvec]).flatten()
    assert len(rvec) == 3, "'rvec' must be a 1D array with 3 values."
    
    # define the rotation class object
    R = Rotation.from_rotvec(rvec)
    
    # get the Euler angles
    return R.as_dcm(), R


# %% FLIP DATA AROUND ITS MEAN VALUE
def flip(X):
    '''
    Flip X around its mean

    Input
        X   :   (1D Array)
                the data

    Output
        F   :   (1D Array)
                X flipped around its mean value
    '''

    # import the necessary packages
    import numpy as np

    # check X is a 1D array
    X = np.atleast_1d(X)
    assert X.ndim == 1, '"X" must be a 1D array.'
    X = np.array([X]).flatten()

    # get the mean
    u = np.mean(X)

    # flip the data around the mean
    return -1 * (X - u) + u



# ROTATION MATRIX EDITOR
def rotmat(angle=0, dimensions=3, axis=1, sympy=False):
    """
    Create a rotation matrix along the i-th axis of an N-dimensional
    vector. The function returns the rotation matrix T which satisfies:
                            R_kn = T_nn(a, i) x M_kn
    where M and R are generic k-by-n matrices and T_nn is the rotation
    matrix that rotates M_kn along the i-th dimension of "a" radiants.

    :param angle: the angle of rotation in radiants.
    :param dimensions: the number of dimensions of the rotation matrix.
    :param axis: the axis along which the rotation have to occur.
    :param sympy: should the resulting rotation matrix be reported as a numpy ndarray or a sympy Matrix? in the
    latter case,
    :return: the ndarray resulting in the desired rotation matrix.
    """

    # import the required packages
    import numpy as np

    # create an empty rotation matrix
    T = np.zeros((dimensions, dimensions))
    for r in np.arange(dimensions) + 1:
        for c in np.arange(dimensions) + 1:

            # set the values of the main diagonal belonging to the fixed
            # axis to 1
            if (r == c):
                T[r - 1, c - 1] = 1 if np.all([axis == r, axis == c]) else np.cos(angle)
            else:

                # manage the elements outside the main diagonal that are
                # not part of the fixed axis
                if not (axis in [r, c]):

                    # lower triangle elements outside the fixed axis are
                    # -sign * sin(a)
                    if r > c:
                        T[r - 1, c - 1] = (-1) * (-1)**(r + c) * np.sin(angle)

                    # upper triangle elements outside the fixed axis are
                    # sign * sin(a).
                    else:
                        T[r - 1, c - 1] = (-1)**(r + c) * np.sin(angle)
    return T


# %% ANGLE CALCULATION FROM A VECTOR
def pitch_and_roll(X, R='XYZ'):
    '''
    get the pitch and roll angles of a 3D accelerometer collected from
    static stance. This function assumes there is no rotation along the 3rd
    (i.e. vertical) axis leading to a Yaw angle = 0

    Input
        X   :   (3-by-n numpy array)
                the accelerometer data with each row reflecting the
                respectively the X, Y, Z axis.

     Input
        R   :   (str)
                the rotation order. By default it is 'XYZ' (i.e. pitch --> roll
                --> yaw rotation order). The other option is 'YXZ' (i.e. roll
                --> pitch  --> yaw rotation order).

    Output
        P   :   (float)
                the resulting pitch angle (i.e. the angle rotating around the
                X axis) in radiants.

        R   :   (float)
                the resulting roll angle (i.e. the angle rotating around the Y
                axis) in radiants. in radiants.

    References
        Pedley M (2013) Tilt Sensing Using a Three-Axis Accelerometer.
            Freescale Semiconductor Application Note. Document Number:
            AN3461, Rev.6, 03/2013. url:
            https://arduino.onepamop.com/wp-content/uploads/2016/03/AN3461.pdf
    '''

    # import the required packages
    import numpy as np

    # check the rotation order
    assert (R.upper() in ['XYZ', 'YXZ'])

    # get the average acceleration in each axis
    Z = np.mean(X, 1)

    # get the pitch and roll angles
    if R == 'XYZ':
        R = np.arctan2(Z[1], Z[2])
        P = np.arctan2(-Z[0], np.sqrt(Z[1] ** 2 + Z[2] ** 2))
    elif R == 'YXZ':
        R = np.arctan2(Z[1], np.sqrt(Z[0] ** 2 + Z[2] ** 2))
        P = np.arctan2(-Z[0], Z[2])

    # return the angles orientation
    return P, R


# %% PERFORM (MULTIPLE) CROSS CORRELATION
def xcorr(X, c_type='unbiased', return_negative=False):
    '''
    set the cross correlation of the data in X

    Input:
                     X : (P x N numpy array)
                         P = the number of variables
                         N = the number of samples

                c_type : str
                         {'biased', 'unbiased'}

        return_negative: bool
                         Should the negative lags be reported?

    Note:
        if X is a 1 x N array or an N length 1-D array, the autocorrelation is
        provided.
    '''
    import numpy as np
    from scipy.signal import fftconvolve as fftxcorr

    # ensure the shape of X
    X = np.atleast_2d(X)

    # take the autocorrelation if X is a 1-d signal
    if X.shape[0] == 1:
        X = np.vstack((X, X))
    P, N = X.shape

    # remove the mean from each dimension
    V = X - np.atleast_2d(np.mean(X, 1)).T

    # take the cross correlation
    xc = []
    for i in np.arange(P - 1):
        for j in np.arange(i + 1, P):

            # FFT convolution
            R = fftxcorr(V[i], V[j][::-1], "full")

            # store the value
            R = np.atleast_2d(R)
            xc = np.vstack((xc, R)) if len(xc) > 0 else np.copy(R)

    # average over all the multiples
    xc = np.mean(xc, 0)

    # adjust the output
    lags = np.arange(-(N - 1), N)
    if not return_negative:
        xc = xc[(N - 1):]
        lags = lags[(N - 1):]

    # normalize
    if c_type == 'unbiased':
        xc /= (N + 1 - abs(lags))
    elif c_type == 'biased':
        xc /= (N + 1)
    else:
        st = 'The "c_type" parameter was not correctly specified. The "biased"'
        st += ' estimator has been used'
        print(st)
        xc /= (N + 1)
    return xc, lags
