
########    IMPORTS    ########



import numpy as np
import pandas as pd



########    CLASSES    ########



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