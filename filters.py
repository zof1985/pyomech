
########    IMPORTS    ########



import numpy as np
import pandas as pd


eps = np.finfo(float).eps



########    CLASSES    ########



class KalmanFilter():
    
    
    def __init__(self, X, P, M, estFUN, R=None, **kwargs):
        """
        Initialize the filter. The general Kalman filtering procedure is a 2 steps approach as
        described below:  

            Prediction: Provide a statistical estimate of where the next measure should be.
        
            Update:     Compare the predicted state with the effective measure and adjust the
                        filter gain.

        Input:

            X:      (Nx1 array)
                    the initial state vector provided as a column vector with length N.
            
            P:      (NxN array)
                    the initial covariance matrix corresponding to the state X.

            M:      (JxN array)
                    a matrix used to convert the "state space" to the "measurement space".
                    This function must accept a column vector of dimension N and a series of
                    key_labelled parameters. It must return a column vector of dimension J.
            
            R:      (JxJ array or None)
                    a column vector defining the noise to be added to the new measurement
                    (e.g. instrumental error) during the calculation of the filter gain.
                    It might be None. In this case it is inizialized to be an array of zeros.
                    
            estFUN: (function)
                    a function having form "function(X, P, **kwargs)". This function must
                    convert the input X (the state vector) and P (its covariance matrix) to
                    their estimates at the next sample time.
                    This function must accept a column vector of dimension N, an NxN matrix
                    and a series of key_labelled parameters. It then must return a column
                    vector of dimension N, an NxN matrix.
        
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
        txt = "'X' must be a (N, 1) numpy.ndarray or pandas.DataFrame instance with 1 column."
        assert isinstance(X, (np.ndarray, pd.DataFrame)) and X.shape[1] == 1, txt
        self.N = X.shape[0]
        self.X = X
        self.I = np.eye(self.N)

        # get P
        txt = "'P' must be a (N, N) numpy.ndarray or pandas.DataFrame instance with N columns"
        txt += " and rows."
        assert isinstance(P, (np.ndarray, pd.DataFrame)), txt
        assert np.all([i == self.N for i in P.shape]), txt
        self.P = P

        # get M and J
        txt = "'M' must be a (J, N) numpy.ndarray or pandas.DataFrame instance with N columns"
        txt += " and J rows."
        assert isinstance(M, (np.ndarray, pd.DataFrame)) and M.shape[1] == self.N, txt
        self.J = M.shape[0]
        self.M = M
        
        # get R
        if R is not None:
            txt = "'R' must be a (J, J) numpy.ndarray or pandas.DataFrame instance with J columns"
            txt += " and rows."
            assert isinstance(R, (np.ndarray, pd.DataFrame)), txt
            assert np.all([i == self.J for i in R.shape]), txt
        else:
            R = np.zeros((self.J, self.J))
        self.R = R

        # get the estFUN and kwargs
        self.estimate = estFUN
        self.est_kwargs = kwargs
    

    def filter(self, Z, R=None):
        """
        filter the new measurement (Z) according to the provided measurment noise (R) without
        updating the current filter state.

        Input:

            Z:  (Jx1 array)
                a new measurement.
            
            R:  (JxJ array or None)
                a new measurement error matrix. If None, the actual store R is used.

        Output:

            Y:  (Jx1 array)
                the residuals between the new measurement and the predicted state.
            
            S:  (JxJ array)
                the innovated covariance of the measurement.
        """

        # return the filtered signal
        return self._filt(Z, R)[0]

    
    def filter_update(self, Z, R=None):
        """
        filter the new measurement (Z) according to the provided measurment noise (R) and
        update the current filter state.

        Input:

            Z:  (Jx1 array)
                a new measurement.
            
            R:  (JxJ array or None)
                a new measurement error matrix. If None, the actual store R is used.

        Output:

            Y:  (Jx1 array)
                the residuals between the new measurement and the predicted state.
            
            S:  (JxJ array)
                the innovated covariance of the measurement.
        """

        # get the filtered data
        Zfil, Xnew, Pnew = self._filt(Z, R)

        # update the state ofthe filter
        self.X = Xnew
        self.P = Pnew

        # return the filtered measure
        return Zfil


    def _filt(self, Z, R=None):
        """
        Internal function that performs all the steps.

        Input:

            Z:      (Jx1 array)
                    a new measurement.
            
            R:      (JxJ array or None)
                    a new measurement error matrix. If None, the actual store R is used.

        Output:

            Zfil:   (Jx1 array)
                    the filtered signal
            
            Xnew:   (Nx1 array)
                    the updated state vector.
            
            Pnew:   (NxN array)
                    the updated covariance matrix.
        """
        
        # check the entries
        txt = "'Z' must be a (J, 1) numpy.ndarray or pandas.DataFrame instance with 1 column"
        txt += " and J rows."
        assert isinstance(Z, (np.ndarray, pd.DataFrame)), txt
        assert Z.shape[1] == 1 and Z.shape[0] == self.J, txt
        if R is not None:
            txt = "'R' must be a (J, J) numpy.ndarray or pandas.DataFrame instance with J columns"
            txt += " and rows."
            assert isinstance(R, (np.ndarray, pd.DataFrame)), txt
            assert np.all([i == self.J for i in R.shape]), txt
            self.R = R

        # get the estimates
        Xest, Pest = self.estimate(self.X, self.P, **self.est_kwargs)
        Xest[np.argwhere(abs(Xest) < eps)] = 0.
        Pest[np.argwhere(abs(Pest) < eps)] = 0.
                
        # innovation phase
        Y = Z - self.M.dot(Xest)
        S = self.M.dot(Pest).dot(self.M.T) + self.R
        S[np.argwhere(abs(S) < eps)] = 0.

        # get the Kalman gain
        K = Pest.dot(self.M.T).dot(np.linalg.pinv(S))
        K[np.argwhere(abs(K) < eps)] = 0.

        # update
        Xnew = Xest + K.dot(Y)
        Pnew = (self.I - K.dot(self.M)).dot(Pest)
        Pnew[np.argwhere(abs(Pnew) < eps)] = 0.

        # finalize
        Znew = self.M.dot(Xnew)
        return Znew, Xnew, Pnew



class KalmanFilterOld():
    
    
    def __init__(self, X, P, F=None, U=None, B=None, Z=None, H=None, Q=None, R=None):
        """
        Initialize the filter. The general Kalman filtering procedure is a 2 steps approach as described below:
                            
                            Prediction steps:


                                X_est = F * X + B * u

                                P_est = F * P * F' + Q


                            Update steps:


                                            P_est * H'
                                    K = ------------------
                                        H * P_est * H' + R

                                X_out = X_est + K * (Z - H * X_est)

                                P_out = (I - K * H) * P_est

    
        Input:               
            X:  (Nx1 array)
                the starting state vector.
        
            U:  (Lx1 array)
                the motion matrix (i.e. the inputs that may affect the transition of the state from one time step to
                the other.
        
            P:  (NxN array)
                the covariance matrix of the state vector.
        
            Z:  (Mx1 array)
                the new measurement.

            F:  (NxN array)
                the state transition function. This matrix moves the "old" state variable (X) to the new state variable
                (X_est).

            B:  (NxL array)
                the motion transition function. This matrix moves the motion (U) to the new state.
        
            H:  (MxN array)
                the measurement function. This matrix converts the state variable (X_est) from the state space to the
                measurement space (i.e. the same of Z).

            R:  (MxM array)
                the measurement noise. This is a diagonal matrix containing the noise of each measure.

            Q:  (NxN array)
                the process noise to be added to the covariance estimate.        
        """

        # get the expected N
        X                    = np.atleast_2d(X)
        if X.shape[1] > 1: X = X.T
        self.N               = X.shape[0]

        # generate the identity matrix
        self.I = np.eye(self.N)

        # check X
        self.__assertShape(X, "X", self.N, 1, False)
        self.X               = X

        # check F
        self.__assertShape(F, "F", self.N, self.N)
        self.F               = F

        # check P
        self.__assertShape(P, "P", self.N, self.N, False)
        self.P               = P

        # get the expected L
        U                    = np.atleast_2d(U)
        if U.shape[1] > 1: U = U.T
        self.L               = U.shape[0]

        # check U
        self.__assertShape(U, "U", self.L, 1)
        self.U               = U

        # check B
        self.__assertShape(B, "B", self.N, self.L)
        self.B               = B
        
        # get the expected M
        Z                    = np.atleast_2d(Z)
        if Z.shape[1] > 1: Z = Z.T
        self.M               = Z.shape[0]

        # check Z
        self.__assertShape(Z, "Z", self.M, 1)
        self.Z               = Z

        # check H
        self.__assertShape(H, "H", self.M, self.N)
        self.H               = H

        # check Q
        self.__assertShape(Q, "Q", self.N, self.N)
        self.Q               = Q

        # check R
        self.__assertShape(R, "R", self.M, self.M)
        self.R               = R
       

    def filter_update(self, Z, F=None, U=None, B=None, H=None, Q=None, R=None):
        """
        filter the new measurement and update the state of the filter.
                            
                            Prediction steps:


                                X_est = F * X + B * u

                                P_est = F * P * F' + Q


                            Update steps:


                                            P_est * H'
                                    K = ------------------
                                        H * P_est * H' + R

                                X_out = X_est + K * (Z - H * X_est)

                                P_out = (I - K * H) * P_est

    
        Input:               
            Z:  (Mx1 array)
                the new measurement.

            U:  (Lx1 array, None)
                the motion matrix (i.e. the inputs that may affect the transition of the state from one time step to
                the other.

            F:  (NxN array, None)
                the state transition function. This matrix moves the "old" state variable (X) to the new state variable
                (X_est).

            B:  (NxL array, None)
                the motion transition function. This matrix moves the motion (U) to the new state.
        
            H:  (MxN array, None)
                the measurement function. This matrix converts the state variable (X_est) from the state space to the
                measurement space (i.e. the same of Z).

            R:  (MxM array, None)
                the measurement noise. This is a diagonal matrix containing the noise of each measure.

            Q:  (NxN array, None)
                the process noise to be added to the covariance estimate.        
        
        Output:
            X:  (row vector)
                the filtered state-space corresponding to the new measurement.
    
            P:  (square matrix)
                the covariance matrix smoothed by the Kalman gain.
        """   
           
        
        # CHECK THE ENTERED PARAMETERS #


        # check F
        if F is not None:
            self.__assertShape(F, "F", self.N, self.N, False)
            self.F               = F

        # get the expected L
        if U is not None:
            U                    = np.atleast_2d(U)
            if U.shape[1] > 1: U = U.T
            self.L               = U.shape[0]

        # check U
        if U is not None:
            self.__assertShape(U, "U", self.L, 1, False)
            self.U               = U

        # check B
        if B is not None:
            self.__assertShape(B, "B", self.N, self.L, False)
            self.B               = B
        
        # get the expected M
        Z                        = np.atleast_2d(Z)
        if Z.shape[1] > 1:   Z   = Z.T
        self.M                   = Z.shape[0]

        # check Z
        self.__assertShape(Z, "Z", self.M, 1, False)
        self.Z                   = Z

        # check H
        if H is not None:
            self.__assertShape(H, "H", self.M, self.N, False)
            self.H               = H

        # check Q
        if Q is not None:
            self.__assertShape(Q, "Q", self.N, self.N, False)
            self.Q               = Q

        # check R
        if R is not None:
            self.__assertShape(R, "R", self.M, self.M, False)
            self.R               = R

        
        # PREDICTION STEP #


        # 1. X_est = F * X + B * u
        #    Here a wild guess of the state corresponding to the next time step is provided.
        self.X = self.F.dot(self.X) + self.B.dot(self.U) 
        
        # 2. P_est = F * P * F' + Q
        #    This step calculates the covariance matrix corresponding to the estimated new state.
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q


        # UPDATE STEP #


        # 3. get the kalman gain
        #    the Kalman gain reflects how confident we are to the predicted output compared to the "measured" output.
        try:

            # this step requires matrix inversion
            self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self.P).dot(self.H.T) + self.R))

        except Exception:
            
            # if the matrix inversion fails, try with the pseudo inverse
            self.K = self.P.dot(self.H.T).dot(np.linalg.pinv(self.H.dot(self.P).dot(self.H.T) + self.R))
           
        # 4. update the state and the covariance
        #    this step corrects X by the effect of the new measurement smoothed by the Kalman gain 
        self.X = self.X + self.K.dot(self.Z - self.H.dot(self.X))
        I_KH = self.I - self.K.dot(self.H)
        self.P = I_KH.dot(self.P).dot(I_KH.T) + self.K.dot(self.R).dot(self.K.T)
    
        # 5. return the new (filtered) state in the measurement space
        return self.H.dot(self.X)


    def __assertShape(self, what, name, rows, cols, none=True):
        """
        check whether the size of the provided variable is consistent with the expectations.

        Input:
            what:   (Object)
                    any variable
            
            name:   (str)
                    the name of the variable
            
            rows:   (int)
                    the number of expected rows
            
            cols:   (int)
                    the number of expected cols
            
            none:   (bool)
                    True means that "what" can be None. False, otherwise
        """
        if what is not None:
            txt = "{} must be a {}x{} array".format(name, rows, cols)
            assert what.shape[0] == rows and what.shape[1] == cols, txt
        else:
            assert isinstance(what, NoneType), "{} cannot be None".format(name)