
########    IMPORTS    ########



import numpy as np
import pandas as pd
from . import vectors as pv
import matplotlib.pyplot as pl



########    METHODS    ########



def rk4(t0, y0, dt, fun, **kwargs):
    """
    Fourth order Runge-Kutta integrator for the Ordinary Differential Equation
    based on discrete data. This method is based on the integration of the function:

                       dy
            f(t, y) = ----
                       dx

    with y0 = y(t0)

    Input:
            
        t0:     (float)
                the time zero of the integration.
            
        y0:     (float)
                the starting value.

        dt:     (float)
                the time step from t0.

        fun:    (function)
                the function being the first derivative of y over t.

        kwargs: (objects)
                additional data passed to f.
    
    Output:

        y:      (1D array)
                the integrated signal over t.
        """
    

    def _Kn(k, fun, dt, y, t, **kwargs):
        """
        get the n-th K value for finding the solution.

        Input:

            k:      (int)
                    the actual K-th.

            fun:    (function)
                    the function returing the first derivative of y.
                
            dt:      (float)
                    the step size.
                
            y:      (float)
                    the actual y value.

            t:      (float)
                    the t value.

            kwargs: (objects)
                    parameters passed to f.
            
        Output:

            y1:     (float)
                    the resulting Y value.
        """

        if k == 1:
            return fun(t, y, **kwargs)
        elif k == 4:
            return fun(t + dt, y + _Kn(k - 1, fun, dt, y, t, **kwargs), **kwargs)
        else:
            return fun(t + dt * 0.5, y + 0.5 * _Kn(k - 1, fun, dt, y, t, **kwargs), **kwargs)
    

    # perform the integration
    y = y0
    for i in range(4):
        y += (2 if i > 0 or i < 3 else 1) / 6 * dt * _Kn(i + 1, fun, dt, y0, t0, **kwargs)
    return y
    


########    CLASSES    ########



class IMU():


    def __init__(self, acc, gyr):
        """
        Inertial Measurement Unit class object.

        Input:

            acc:    (pyomech.Vector)
                    accelerometer data.
            
            gyr:    (pyomech.Vector)
                    gyroscope data.
        """
        
        # check the entries
        txt = "{} must be a pyomech.Vector instance."
        assert isinstance(acc, pv.Vector), txt.format("acc")
        self.accelerometer = acc
        assert isinstance(gyr, pv.Vector), txt.format("gyr")
        self.gyroscope = gyr / 180 * np.pi  # convert to rad/s
    

    def madgwickPose(self, beta=0.06544985):
        """
        estimate the pose of the IMU using a gyroscope and accelerometer.
        
        Input:

            beta:   (float)
                    the gain to be applied for compensating the gyro measurement error.
                    By default it is set at ~5°/sec, although it is provided in rad.
        """

        # initialize
        A = self.accelerometer / self.accelerometer.module.values  # normalized accelerations
        W = self.gyroscope.copy()  # it must be provided in rad/s
        SEq = np.quaternion(1, 0, 0, 0)  # quaternion defining the orientation initiali conditions

        # iterate the estimation
        for n, i in enumerate(A.index.to_numpy()):
            dt = (1 / A.sampling_frequency) if n == 0 else (i - A.index.to_numpy()[n - 1]) # get dt

            # get the objective function
            F = []
"""
// Local system variables
float norm; // vector norm
float SEqDot_omega_1, SEqDot_omega_2, SEqDot_omega_3, SEqDot_omega_4; // quaternion derrivative from gyroscopes elements
float f_1, f_2, f_3; // objective function elements
float J_11or24, J_12or23, J_13or22, J_14or21, J_32, J_33; // objective function Jacobian elements
float SEqHatDot_1, SEqHatDot_2, SEqHatDot_3, SEqHatDot_4; // estimated direction of the gyroscope error
// Axulirary variables to avoid reapeated calcualtions
float halfSEq_1 = 0.5f * SEq_1;
float halfSEq_2 = 0.5f * SEq_2;
float halfSEq_3 = 0.5f * SEq_3;
float halfSEq_4 = 0.5f * SEq_4;
float twoSEq_1 = 2.0f * SEq_1;
float twoSEq_2 = 2.0f * SEq_2;
float twoSEq_3 = 2.0f * SEq_3;
// Normalise the accelerometer measurement
norm = sqrt(a_x * a_x + a_y * a_y + a_z * a_z);
a_x /= norm;
a_y /= norm;
a_z /= norm;
// Compute the objective function and Jacobian
f_1 = twoSEq_2 * SEq_4 - twoSEq_1 * SEq_3 - a_x;
f_2 = twoSEq_1 * SEq_2 + twoSEq_3 * SEq_4 - a_y;
f_3 = 1.0f - twoSEq_2 * SEq_2 - twoSEq_3 * SEq_3 - a_z;
J_11or24 = twoSEq_3; // J_11 negated in matrix multiplication
J_12or23 = 2.0f * SEq_4;
J_13or22 = twoSEq_1; // J_12 negated in matrix multiplication
J_14or21 = twoSEq_2;
J_32 = 2.0f * J_14or21; // negated in matrix multiplication
J_33 = 2.0f * J_11or24; // negated in matrix multiplication
// Compute the gradient (matrix multiplication)
SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1;
SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3;
SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1;
SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2;
// Normalise the gradient
norm = sqrt(SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4);
SEqHatDot_1 /= norm;
SEqHatDot_2 /= norm;
SEqHatDot_3 /= norm;
SEqHatDot_4 /= norm;
// Compute the quaternion derrivative measured by gyroscopes
SEqDot_omega_1 = -halfSEq_2 * w_x - halfSEq_3 * w_y - halfSEq_4 * w_z;
SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z - halfSEq_4 * w_y;
SEqDot_omega_3 = halfSEq_1 * w_y - halfSEq_2 * w_z + halfSEq_4 * w_x;
SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y - halfSEq_3 * w_x;
// Compute then integrate the estimated quaternion derrivative
SEq_1 += (SEqDot_omega_1 - (beta * SEqHatDot_1)) * deltat;
SEq_2 += (SEqDot_omega_2 - (beta * SEqHatDot_2)) * deltat;
SEq_3 += (SEqDot_omega_3 - (beta * SEqHatDot_3)) * deltat;
SEq_4 += (SEqDot_omega_4 - (beta * SEqHatDot_4)) * deltat;
// Normalise quaternion
norm = sqrt(SEq_1 * SEq_1 + SEq_2 * SEq_2 + SEq_3 * SEq_3 + SEq_4 * SEq_4);
SEq_1 /= norm;
SEq_2 /= norm;
SEq_3 /= norm;
SEq_4 /= norm;
}

        """

    def integrateTime(self, V, S, dim_unit, type):
        """
        generate a Kalman filter and use it to obtain integrated data.

        Input:

            V:  (pyomech.Vector)
                the vector containing the data to be integrated.
            
            S:  (pyomech.Vector)
                the vector containing the static data used to extract sensor/activity
                measurement noise.
        
        Output:

            X:  (pyomech.Vector)
                A vector with the same dimensions of V integrated over time.
        """

        # get the initial state
        X = np.zeros((V.shape[1], 1))

        # get the measurement to state matrix
        B = np.eye(V.shape[1]) / V.sampling_frequency

        # get the state to measurement
        H = np.eye(V.shape[1])

        # initial state variance
        P = S.cov().values

        # matrix used to estimate the new state.
        F = np.eye(V.shape[1])

        # process noise variance
        Q = np.copy(P)
        
        # get the measurement noise
        U = S.var().values
        R = np.copy(P)

        # use the filter to process the data
        KF = KalmanFilter(X, P, H, F, B, U, Q, R)
        K = zip(V.index.to_numpy()[:-1], V.index.to_numpy()[1:])
        Z = np.vstack([KF.filter_update(V.loc[v].values - V.loc[i].values).T for i, v in K])
        
        # get the data
        return pv.Vector(Z, columns=V.columns, index=V.index[:-1], dim_unit=dim_unit,
                         time_unit=V.time_unit, type=type)


    def doubleIntegrateTime(self, V, S, dim_unit, type):
        """
        generate a Kalman filter and use it to obtain double integrated data over time.

        Input:

            V:  (pyomech.Vector)
                the vector containing the data to be integrated.
            
            S:  (pyomech.Vector)
                the vector containing the static data used to extract sensor/activity
                measurement noise.
        
        Output:

            X:  (pyomech.Vector)
                A vector with the same dimensions of V integrated over time.
        """

        # get dt
        dt = 1 /  V.sampling_frequency

        # get the initial state
        X = np.zeros((V.shape[1] * 2, 1))

        # get the measurement to state matrix
        B = []
        for i in np.arange(V.shape[1]):
            L = [0 for j in np.arange(V.shape[1])]
            B += [L]
            L = [0 for j in np.arange(V.shape[1])]
            L[i] = dt
            B += [L]
        B = np.atleast_2d(B)
        
        # get the state to measurement
        H = []
        for i in np.arange(V.shape[1]):
            L = [0 for j in np.arange(V.shape[1] * 2)]
            L[i * 2 + 1] = 1
            H += [L]
        H = np.atleast_2d(H)

        # matrix used to estimate the new state.
        F = []
        for i in np.arange(V.shape[1]):
            L = [0 for j in np.arange(V.shape[1] * 2)]
            L[2 * i] = 1
            L[2 * i + 1] = dt
            F += [L]
            L = [1 if j == (i * 2 + 1) else 0 for j in np.arange(V.shape[1] * 2)]
            F += [L]
        F = np.atleast_2d(F)

        # initial state variance
        P = H.T.dot(S.cov().values).dot(H)

        # process noise variance
        Q = np.copy(P)
        
        # get the measurement noise
        U = S.var().values
        R = S.cov().values

        # conversion matrix
        K = []
        for i in np.arange(V.shape[1]):
            L = [0 for j in np.arange(X.shape[0])]
            L[i * 2] = 1
            K += [L]
        K = np.atleast_2d(K)

        # use the filter to process the data
        KF = KalmanFilter(X, P, H, F, B, U, Q, R)
        L = zip(V.index.to_numpy()[:-1], V.index.to_numpy()[1:])
        Z = np.vstack([K.dot(KF.filter_update(V.loc[v].values - V.loc[i].values)).T for i, v in L])
        
        # get the data
        return pv.Vector(Z, columns=V.columns, index=V.index[:-1], dim_unit=dim_unit,
                         time_unit=V.time_unit, type=type)


    def _show(self, V):
        """
        shortcut function used to plot data in V.

        Input:

            V:  (pyomech.Vector)
                vector of data to be plotted.
        """
        
        idx = V.index.to_numpy()
        for c in V.columns.to_numpy():
            pl.plot(idx, V[c].values.flatten(), label=c)
        pl.plot(idx, np.zeros((len(idx), 1)).flatten(), 'k--')
        pl.legend()
        pl.show()



class KalmanFilter():
    
    
    def __init__(self, x, P, H, F, B, u=None, Q=None, R=None):
        """
        Initialize the filter. The general Kalman filtering procedure is a 2 steps approach as
        described below:  

            Prediction: Provide a statistical estimate of where the next measure should be.
        
            Update:     Compare the predicted state with the effective measure and adjust the
                        filter gain.

        Input:

            x:      (Nx1 array)
                    the initial state vector provided as a column vector with length N.
            
            P:      (NxN array)
                    the initial covariance matrix corresponding to the state X.

            H:      (JxN array)
                    a matrix used to convert the "state space" to the "measurement space".
                    This function must accept a column vector of dimension N and a series of
                    key_labelled parameters. It must return a column vector of dimension J.
                    
            F:      (NxN array)
                    the transition matrix from the actual state to the next.
            
            B:      (NxJ array)
                    the transition matrix from the measurement to the state space.

            u:      (Nx1 array or None)
                    a vector containing the process noise, i.e. the noise to be added at each
                    new update to the new state estimate.
                    It might be None. In this case it is inizialized to be an array of zeros.

            R:      (JxJ array or None)
                    a square matrix defining the measurement noise to be added to each new
                    measurement (e.g. instrumental error) during the calculation of the filter
                    gain. It might be None. In this case it is inizialized to be an array of zeros.

            Q:      (NxN array or None)
                    a square matrix defining the noise to be added to the covariance of the state
                    space (e.g. instrumental error) during the calculation of the filter gain.
                    It might be None. In this case it is inizialized to be an array of zeros.


        Action of the filter:

            The filter contains 2 main functions: "filter" and "filter_update". Their action is
            the following:

            1)  The filter uses the provided "estimate" function to predict the next state:

                                   Xest, Pest = estFUN(X, P, **kwargs)
                
            2)  Next, the "innovation" phase is obtained measuring the difference between the
                estimated new state (Xest, Pest) and a new measurement (Z):

                                            Y = Z - M x Xest
                                            S = M x Pest x M' + R
                
            3)  Afterwards, the "update" phase adjusts the filter gain and corrects the former
                estimate and covariance.

                                            K = Pest x M' x (S ** -1)
                                         Xnew = Xest + K x Y
                                         Pnew = (I - K x M) x Pest
                
            4)  If the "filter_update" function was called, the updated state and covariance are
                stored as new filter state and covariance values.
            
                                            X = Xnew
                                            P = Pnew
            
            5)  The filter returns the filtered measure.

                                         Zfil = M x Xnew        
        """

        # get X, N and I
        assert isinstance(x, (np.ndarray, pd.DataFrame)) and x.shape[1] == 1, self.msg("x", "N", 1)
        self.N = x.shape[0]
        self.x = x

        # get P
        assert isinstance(P, (np.ndarray, pd.DataFrame)), self.msg("P", self.N, self.N)
        assert np.all([i == self.N for i in P.shape]), self.msg("P", self.N, self.N)
        self.P = P

        # get H and J
        txt = self.msg("H", "J", self.N)
        assert isinstance(H, (np.ndarray, pd.DataFrame)) and H.shape[1] == self.N, txt
        self.J = H.shape[0]
        self.H = H
        
        # get F
        txt = self.msg("F", self.N, self.N)
        assert isinstance(F, (np.ndarray, pd.DataFrame)), txt
        assert np.all([i == self.N for i in F.shape]), self.msg("F", self.N, self.N)
        self.F = F

        # get B
        assert isinstance(B, (np.ndarray, pd.DataFrame)), self.msg("B", self.N, self.J)
        assert B.shape[1] == self.J and B.shape[0] == self.N, self.msg("B", self.N, self.J)
        self.B = B

        # get u
        if u is not None:
            assert isinstance(u, (np.ndarray, pd.DataFrame)), self.msg("u", self.J, 1)
            assert u.shape[0] == self.J and u.shape[1] == 1, self.msg("u", self.J, 1)
        else:
            u = np.zeros((self.J, 1))
        self.u = u

        # get R
        if R is not None:
            assert isinstance(R, (np.ndarray, pd.DataFrame)), self.msg("R", self.J, self.J)
            assert np.all([i == self.J for i in R.shape]), self.msg("R", self.J, self.J)
        else:
            R = np.zeros((self.J, self.J))
        self.R = R

        # get Q
        if Q is not None:
            assert isinstance(Q, (np.ndarray, pd.DataFrame)), self.msg("Q", self.N, self.N)
            assert np.all([i == self.N for i in Q.shape]), self.msg("Q", self.N, self.N)
        else:
            Q = np.zeros((self.N, self.N))
        self.Q = Q


    def msg(self, X, N, M):
            txt = "'{}' must be a ({}, {}) numpy.ndarray or pandas.DataFrame instance with "
            txt += "{} rows and {} columns."
            return txt.format(X, N, M, N, M)    


    def filter(self, z, u=None, Q=None, R=None):
        """
        filter the new measurement (Z) according to the provided noise (u, R and Q) without
        updating the current filter state.

        Input:

            z:  (Jx1 array)
                a new measurement.
            
            u:  (Nx1 array or None)
                a vector containing the process noise, i.e. the noise to be added at each
                new update to the new state estimate.
                It might be None. In this case it is inizialized to be an array of zeros.

            R:  (JxJ array or None)
                a square matrix defining the measurement noise to be added to each new
                measurement (e.g. instrumental error) during the calculation of the filter
                gain. It might be None. In this case it is inizialized to be an array of zeros.

            Q:  (NxN array or None)
                a square matrix defining the noise to be added to the covariance of the state
                space (e.g. instrumental error) during the calculation of the filter gain.
                It might be None. In this case it is inizialized to be an array of zeros.

        Output:

            Y:  (Jx1 array)
                the residuals between the new measurement and the predicted state.
            
            S:  (JxJ array)
                the innovated covariance of the measurement.
        """

        # return the filtered signal
        return self._filt(z, u, R, Q)[0]

    
    def filter_update(self, z, u=None, Q=None, R=None):
        """
        filter the new measurement (Z) according to the provided noise (u, R and Q) and
        update the current filter state.

        Input:

            z:  (Jx1 array)
                a new measurement.
                        
            u:  (Nx1 array or None)
                a vector containing the process noise, i.e. the noise to be added at each
                new update to the new state estimate.
                It might be None. In this case it is inizialized to be an array of zeros.

            R:  (JxJ array or None)
                a square matrix defining the measurement noise to be added to each new
                measurement (e.g. instrumental error) during the calculation of the filter
                gain. It might be None. In this case it is inizialized to be an array of zeros.

            Q:  (NxN array or None)
                a square matrix defining the noise to be added to the covariance of the state
                space (e.g. instrumental error) during the calculation of the filter gain.
                It might be None. In this case it is inizialized to be an array of zeros.

        Output:

            Y:  (Jx1 array)
                the residuals between the new measurement and the predicted state.
            
            S:  (JxJ array)
                the innovated covariance of the measurement.
        """

        # get the filtered data
        Xnew, Pnew = self._filt(z, u, R, Q)

        # update the state ofthe filter
        self.x = Xnew
        self.P = Pnew

        # return the filtered measure
        return Xnew


    def _filt(self, z, u=None, R=None, Q=None):
        """
        Internal function that performs all the steps.
        """
        
        # check the entries
        assert isinstance(z, (np.ndarray, pd.DataFrame)), self.msg('z', self.J, 1)
        assert z.shape[1] == 1 and z.shape[0] == self.J, self.msg('z', self.J, 1)
        if u is not None:
            assert isinstance(u, (np.ndarray, pd.DataFrame)), self.msg('u', self.J, 1)
            assert u.shape[1] == 1 and u.shape[0] == self.J, self.msg('u', self.J, 1)
            self.u = u
        if R is not None:
            assert isinstance(R, (np.ndarray, pd.DataFrame)), self.msg('R', self.J, self.J)
            assert np.all([i == self.J for i in R.shape]), self.msg('R', self.J, self.J)
            self.R = R
        if Q is not None:
            assert isinstance(Q, (np.ndarray, pd.DataFrame)), self.msg('Q', self.N, self.N)
            assert np.all([i == self.J for i in Q.shape]), self.msg('Q', self.N, self.N)
            self.Q = Q

        eps = np.finfo(float).eps

        # get the estimates
        Xest = self.F.dot(self.x) + self.B.dot(self.u)
        Pest = self.F.dot(self.P).dot(self.F.T) + self.Q
        Xest[np.argwhere(abs(Xest) < eps)] = 0.
        Pest[np.argwhere(abs(Pest) < eps)] = 0.
        
        # get the Kalman gain
        K = Pest.dot(self.H.T).dot(np.linalg.pinv(self.H.dot(Pest).dot(self.H.T) + self.R))
        K[np.argwhere(abs(K) < eps)] = 0.

        # update
        Xnew = Xest + K.dot(z - self.H.dot(Xest))
        Pnew = (np.eye(Pest.shape[1]) - K.dot(self.H)).dot(Pest)
        Xnew[np.argwhere(abs(Xnew) < eps)] = 0.
        Pnew[np.argwhere(abs(Pnew) < eps)] = 0.

        # finalize
        return Xnew, Pnew