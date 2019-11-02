# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:13:43 2018

@author: lzoffoli
"""


# %% SYMMETRY INDEX
def sym_index(A, B):
    '''
    Return the Symmetry Index (SI) as defined by (Robinson et al, 1987):

                         2 * (X_a - X_b)
                SI (%) = --------------- * 100
                            X_a + X_b

    Input
        A, B:   (float)
                the values to be compared

    Output
        SI  :   (float)
                The symmetry Index.

    References:
        Robinson RO, Herzog W, Nigg BM. (1987) Use of force platform variables
            to quantify the effects of chiropractic manipulation on gait
            symmetry. Journal of Manipulative and Physiological Therapeutics,
            10(4), 172–176.
    '''

    # return the symmetry index
    return 2 * abs(A - B) / (A + B) * 100


# %% SYMMETRY ANGLE
def sym_angle(A, B):
    '''
    Return the Symmetry Angle (SA) as defined by Zifchock et al. 2008:

                      pi / 4 - arctan(X_left / X_right)
             SA (%) = --------------------------------- * 100
                                  pi / 2

    Input
        A, B:   (float)
                the values to be compared

    Output
        SA  :   (float)
                The symmetry angle.

    References:
        Zifchock RA, Davis I, Higginson J, Royer T. (2008) The symmetry angle:
            a novel, robust method of quantifying asymmetry. Gait & Posture,
            27 (4), 622–627.
    '''

    # import the necessary packages
    from numpy import pi, arctan, sign

    # get the arctan value
    T = arctan(B / A)

    # get its sign
    S = sign(T)

    # return the index
    return 200. * S * (pi / 4 - abs(T)) / pi


# %% NIGG ET AL (2013) SYMMETRY INDEX
def sym_nigg(A, B, N=101):
    '''
    Return the Symmetry Index accoding to Nigg et al (2013).

    Input
        A, B:   (float)
                the values to be compared

    Input (optional)
        N   :   (int)
                the number of points to be used to interpolate A and B.
                Please note that a cubic spline interpolation will be
                performed.

    Output
        SI  :   (float)
                The symmetry index.

    References:
        Nigg S, Vienneau J, Maurer C, Nigg BM. Development of a symmetry index
            using discrete variables. (2013) Gait Posture. 38(1):115–9.
    '''

    # import the necessary packages
    from numpy import min, max, trapz

    # get the symmetry index
    return trapz(abs(interpolate(A, N) - interpolate(B, N)) * 2. /
                 (max(A) - min(A) + max(B) - min(B)))


# %% SYMMETRY MEASURED AS CORRELATION
def sym_corr(A, B, N=None):
    '''
    Return the Symmetry Index as measure of correlation.

    Input
        A, B:   (float)
                the values to be compared

    Input (optional)
        N   :   (int)
                the number of points to be used to interpolate A and B.
                Please note that a cubic spline interpolation will be
                performed. by default no interpolation is performed.

    Output
        SI  :   (float)
                The symmetry index.
    '''

    # import the necessary packages
    import numpy as np
    import GaitLab.processing as pr

    # check N
    N = len(A) if N is None else N

    # get the interpolated A and B
    Ai = np.atleast_2d(pr.interpolate(A, N))
    Bi = np.atleast_2d(pr.interpolate(B, N))

    # get the symmetry index
    return np.corrcoef(np.vstack((Ai, Bi)))[0, 1] * 100


# %% SYMMETRY INDEX
def sym_ratio(A, B):
    '''
    Return the Symmetry Ratio (SR) as defined:

                SR (%) = (1 - A / B) * 100

    Input
        A, B:   (float)
                the values to be compared

    Output
        SR  :   (float)
                The symmetry Ratio.
    '''

    # return the symmetry index
    return (1 - A / B) * 100
