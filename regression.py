


# IMPORTS



import numpy as np
import pandas as pd
import itertools as it
import scipy.signal as ss
import scipy.interpolate as si
import scipy.linalg as sl
import warnings
from bokeh.plotting import *
from bokeh.layouts import *
from bokeh.models import *



# GENERAL OPTIONS
fig_size = 300  # pixels



# CLASSES



class LinearRegression():



    def __init__(self, y, x, fit_intercept=True):
        """
        Obtain the regression coefficients according to the Ordinary Least Squares approach.

        Input:
            y:              (2D column numpy array)
                            the array containing the dependent variable. 

            x:              (2D array)
                            the array containing the indipendent variables. The number of rows must
                            equal the rows of y, while each column will be a regressor (i.e. an
                            indipendent variable).

            fit_intercept:  (bool)
                            Should the intercept be included in the model? Otherwise it will be
                            set to zero.
        """ 

        # add the input parameters
        self.fit_intercept = fit_intercept

        # correct the shape of y and x
        YY = self.__simplify__(y, 'Y')
        XX = self.__simplify__(x, 'X')
        assert XX.shape[0] == YY.shape[0], "'X' and 'Y' number of rows must be identical."
        
        # add the ones for the intercept
        if self.fit_intercept:
            XX = np.hstack([np.ones((XX.shape[0], 1)), XX])
        
        # get the coefficients and intercept
        self._coefs = pd.DataFrame(sl.inv(XX.T.dot(XX)).dot(XX.T).dot(self.Y),
                                   index=self.__IV_labels__, columns=self.__DV_labels__)



    @property
    def coefs(self):
        """
        vector of the regression coefficients.
        """
        return self._coefs



    def SSPE(self):
        """
        return the Sum of Square Product Error matrix
        """
        R = self.residuals()
        return R.T.dot(R)



    def cov_unscaled(self):
        """
        return the unscaled covariance (i.e. without multiplication for the variance term)
        of the coefficients.
        """
        if self.fit_intercept:
            I = pd.DataFrame({'Intercept': np.tile(1, self.X.shape[0])}, index=self.X.index)
            X = pd.concat([I, self.X], axis=1)
        else:
            X = self.X
        return pd.DataFrame(sl.inv(X.T.dot(X)), index=X.columns, columns=X.columns)



    def residuals(self):
        """
        obtain the residuals of the current regression model.
        """
        return self.Y - self.predict(self.X)



    @property
    def __IV_labels__(self):
        """
        return the labels for the regressors.
        """
        if isinstance(self.X, pd.DataFrame):
            out = []
            if self.fit_intercept:
                out += ['Intercept']
            return out + self.X.columns.to_numpy().tolist()
        elif self.X is None:
            self.X = self.__simplify__(self.X, 'x')
        return ['X{}'.format(i) for i in np.arange(self.X.shape[1] + (1 if self.fit_intercept else 0))]
    


    @property
    def __DV_labels__(self):
        """
        return the labels for the dependent variables.
        """
        if isinstance(self.Y, pd.DataFrame):
            return self.Y.columns.to_numpy().tolist()
        return ['Y{}'.format(i) for i in np.arange(Y.shape[1])]



    def __simplify__(self, v, name):
        """
        internal method to check entries in the constructor.
        """
        txt = "'{}' must be a pandas.DataFrame, a numpy.ndarray or None.".format(name)
        if isinstance(v, pd.DataFrame):
            XX = v
        else:
            if isinstance(v, np.ndarray):
                XX = pd.DataFrame(np.squeeze(v))
            elif v is None:

                # try to get the shape from the other parameter
                try:
                    N = self.Y.shape[0]
                except Exception:
                    try:
                        N = self.X.shape[0]
                    except Exception:
                        raise ValueError(txt)
                XX = np.atleast_2d([[] for i in np.arange(N)])

                # try to convert to the most similar pandas.DataFrame
                # to the other parameter.
                try:
                    IX = self.Y.index
                except Exception:
                    try:
                        IX = self.X.index
                    except Exception:
                        IX = pd.Index(np.arange(N))
                CX = pd.Index(name + "{}".format(i) for i in np.arange(XX.shape[1]) + 1)
                XX = pd.DataFrame(XX, index=IX, columns=CX)
            else:
                raise NotImplementedError(txt)
        setattr(self, name, XX)
        return XX.values



    def copy(self):
        """
        copy the current instance
        """
        return LinearRegression(self.y, self.x, self.digits, self.fit_intercept)



    def predict(self, x):
        """
        predict the fitted Y value according to the provided x.
        """
        if len(self.coefs) == 0:
            return self.intercept
        else:
            X = self.__simplify__(x, 'x')
            n = self.coefs.shape[0] - 1
            assert X.shape[1] == n, "'X' must have {} columns.".format(n)
            Z = X.dot(self.coefs.values[1:]) + self.coefs.values[0]
            if isinstance(x, pd.DataFrame):
                idx = x.index
            else:
                idx = np.arange(X.shape[0])
            return pd.DataFrame(Z, index=idx, columns=self.__DV_labels__)



    def __repr__(self):
        return self.__str__()



    def __str__(self):
        t = "Y = " + (str(self.intercept) if self.fit_intercept != 0 else "")
        for i in np.arange(len(self.coefs)):
            t += (" + " if self.coefs[i] >= 0 else " - ") + str(abs(self.coefs[i]))
            t +=  " X" + str(i + 1)
        return t



    def DF(self):
        """
        return the degrees of freedom of the model.
        """
        return self.Y.shape[0] - self.coefs.shape[0]



    def SS(self):
        """
        calculate the sum of squares of the fitted model.
        """
        P = self.predict(self.X).flatten()
        return np.sum((P - np.mean(P, 0)) ** 2, 0)



    def R2(self):
        """
        calculate the R-squared of the fitted model.
        """
        return self.SS / np.sum((self.Y - np.mean(self.Y, 0)) ** 2, 0)



    def R2_adjusted(self):
        """
        calculate the Adjusted R-squared of the fitted model.
        """
        return 1 - (1 - self.R2) * (self.Y.shape[0] - 1) / (len(self.Y) - self.DF() - 1)



    def RMSE(self):
        """
        Get the Root Mean Squared Error
        """
        return np.sqrt(np.mean(self.residuals() ** 2, 0))



class PolynomialRegression(LinearRegression):


    def __init__(self, y, x, order=None, fit_intercept=True):
        """
        Perform the polynomial regression of y given x. The polynomial order can be specified via order, otherwise the
        order ensuring that all the extracted coefficients are not zero.

        Input:
            y:              (1D array)
                            the array containing the dependent variable.

            x:              (1D array)
                            the array containing the indipendent variable.

            order:          (None or int)
                            the order of the polynomial regression. If None, the order resulting in the highest 
                            Adjusted R-squared.

            fit_intercept:  (bool)
                            Should the intercept be included in the model?
        """ 

        # store the entered parameters
        self.y = np.atleast_2d(y)
        self.x = np.atleast_2d(x)
        if self.x.shape[1] > 1:
            if self.x.shape[0] > 1:
                raise ValueError("'x' must be a 1D array or a 2D array with 1 dimension.")
            else:
                self.x = self.x.T
        self.fit_intercept = fit_intercept
        
        # handle the case the order is not provided
        if order is None:
            order = 0
            r2_adj = -1
            while True:
                self.__fit__(order + 1)
                if np.max([0, self.R2_adjusted]) > r2_adj and self.R2_adjusted < 1:
                    r2_adj = np.max([0, self.R2_adjusted])
                    order += 1
                else:
                    self.__fit__(order)
                    break

        # now handle the case the order is zero
        elif order == 0:
            self.order = order
            X = np.tile(np.mean(y), len(y))
            super(PolynomialRegression, self).__init__(self.y, X, self.fit_intercept)

        # fit the model with the given order
        else:
            self.__fit__(order)



    def __fit__(self, order):
        """
        It returns the set of coefficients corresponding to a model with the provided order.
        """
        self.order = order
        X = np.atleast_2d([self.x ** i for i in np.arange(self.order) + 1]).T
        super(PolynomialRegression, self).__init__(self.y, X, self.fit_intercept)
        self.x = X[:, 0]


    
    def copy(self):
        """
        copy the current instance
        """
        return PolynomialRegression(self.y, self.x, self.order, self.digits, self.fit_intercept)



    def predict(self, x):
        """
        predict the fitted Y value according to the provided x.
        """
        s = np.hstack([(np.atleast_2d(x).T ** (i + 1)) for i in np.arange(self.order)])
        return super(PolynomialRegression, self).predict(s)



    @property
    def R2(self):
        """
        calculate the R-squared of the fitted model.
        """
        if self.predict(self.x) is None:
            check = 1
        return np.corrcoef(self.y, self.predict(self.x))[0, 1] ** 2



    @property
    def R2_adjusted(self):
        """
        calculate the Adjusted R-squared of the fitted model.
        """
        return 1 - (1 - self.R2) * (len(self.y) - 1) / (len(self.y) - self.order - 1)



    @property
    def RMSE(self):
        """
        Get the Root Mean Squared Error
        """
        return np.sqrt(np.mean((self.y - self.predict(self.x)) ** 2))



class PowerRegression():


    def __init__(self, y, x, digits=5):
        """
        Perform the power regression of y given x according to the model:
        
                            y = alpha + beta * (x + gamma) ** delta

        Input:
            y:              (1D array)
                            the array containing the dependent variable.

            x:              (1D array)
                            the array containing the indipendent variable.

            digits:         (int)
                            the number of digits available for each coefficient.
        """ 

        # check the entered data
        assert y.ndim == 1, "'y' must be a 1D array."
        assert x.ndim == 1, "'x' must ba a 1D array."

        # store the entered parameters
        self.y = y
        self.x = x
        self.digits = digits
        eps = np.finfo(float).eps
        
        # get delta as the offset of y
        self.alpha = np.round(np.min(self.y) - 1, digits)
        
        # get Y (ensure no values are zero)
        Y = np.log(np.atleast_2d(y - self.alpha).T)
                
        # get beta as the offset of x
        self.gamma = np.round(1 - np.round(np.min(x), digits), digits)
        
        # get X (ensure no values are zero)
        X = np.hstack([np.ones(Y.shape), np.log(np.atleast_2d(x + self.gamma).T)])
                
        # get the coefficients
        coefs = sl.inv(X.T.dot(X)).dot(X.T).dot(Y)
        
        # get the coefficients
        self.delta = np.round(coefs[1][0], digits)
        self.beta = np.round(np.e ** coefs[0][0], digits)



    def copy(self):
        """
        copy the current instance
        """
        return PowerRegression(self.y, self.x, self.digits)



    def predict(self, x):
        """
        predict use the fitted model to predict regression output according to the model:
        
                            y = alpha + beta * (x + gamma) ** delta.
        """
        return self.alpha + self.beta * (x + self.gamma) ** self.delta



    def __repr__(self):
        return self.__str__()



    def __str__(self):
        return "Y = {:+}{:+}*(X{:+})**({:+})".format(self.alpha, self.beta, self.gamma, self.delta)



    @property
    def R2(self):
        """
        calculate the R-squared of the fitted model.
        """
        return np.corrcoef(self.y, self.predict(self.x))[0, 1] ** 2



    @property
    def R2_adjusted(self):
        """
        calculate the Adjusted R-squared of the fitted model.
        """
        return 1 - (1 - self.R2) * (len(self.y) - 1) / (len(self.y) - self.order - 1)



    @property
    def RMSE(self):
        """
        Get the Root Mean Squared Error
        """
        return np.sqrt(np.mean((self.y - self.predict(self.x)) ** 2))



    def plot_data(self):
        """
        return a bokeh figure representing the data and the regression line.
        """

        # this figure can be generated only if x has 1 dimension
        assert self.x.shape[1] == 1, "x must have 1 dimension."

        # create the figure
        p = figure(width=fig_size, height=fig_size, title="Model fit")

        # print the data
        p.scatter(self.x.flatten(), self.y.flatten(), size=2, fill_color="navy", line_color="navy",
                  marker="circle", fill_alpha=0.5, line_alpha=0.5)
        
        # print the fitting line
        q = np.unique(self.x.flatten())
        z = self.predict(q)
        p.scatter(q, z, size=6, fill_color="darkred", line_color="darkred", marker="circle", fill_alpha=0.5,
                  line_alpha=0.5)
        p.line(q, z, line_color="darkred", line_dash="dashed", line_alpha=0.5, line_width=2)

        # decorate the figure
        self.__plot_layout__(p)

        # return the figure
        return p



    def plot_fit(self):
        """
        return a bokeh figure representing the regression fit of the model.
        """

        # this figure can be generated only if x has 1 dimension
        assert self.x.shape[1] == 1, "x must have 1 dimension."

        # create the figure
        p = figure(width=fig_size, height=fig_size, title="Model fit")

        # print the data
        p.scatter(self.x.flatten(), self.y.flatten(), size=2, fill_color="navy", line_color="navy",
                  marker="circle", fill_alpha=0.5, line_alpha=0.5)
                
        # decorate the figure
        self.__plot_layout__(p)

        # return the figure
        return p



    def __plot_layout__(self, figure):
        """
        internal function used by plot_data and plot_fit to decorate the figure before returning it.
        """

        '''
        # print the model fit
        text = [self.__str__(), "RMSE: {:0.5f}".format(self.RMSE),
                'R-squared adj: {:0.2f}'.format(self.R2_adjusted)]
        for i, t in enumerate(text):
            figure.add_layout(Label(x=np.min(self.x), y=np.max(self.y) * (0.95 - i * 0.1), text=t,
                                    border_line_alpha=0.0, background_fill_alpha=0.7,
                                    background_fill_color="lightyellow", text_font_size="6pt"))
        '''

        # edit the grids
        figure.xgrid.grid_line_alpha=0.3
        figure.ygrid.grid_line_alpha=0.3
        figure.xgrid.grid_line_dash=[5, 5]
        figure.ygrid.grid_line_dash=[5, 5]

        # set the axis labels
        figure.xaxis.axis_label = "X"
        figure.yaxis.axis_label = "Y"