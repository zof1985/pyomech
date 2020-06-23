


# IMPORTS



import numpy as np
import pandas as pd
import itertools as it
import scipy.signal as ss
import scipy.interpolate as si
import warnings
from bokeh.plotting import *
from bokeh.layouts import *
from bokeh.models import *



# GENERAL OPTIONS
fig_size = 300  # pixels



# CLASSES



class LinearRegression():



    def __init__(self, y, x, digits=5, fit_intercept=True):
        """
        Obtain the regression coefficients according to the Ordinary Least Squares approach.

        Input:
            y:              (2D column numpy array)
                            the array containing the dependent variable. 

            x:              (2D array)
                            the array containing the indipendent variables. The number of rows must equal the rows of
                            y, while each column will be a regressor (i.e. an indipendent variable).

            digits:         (int)
                            the number of digits available for each coefficient.

            fit_intercept:  (bool)
                            Should the intercept be included in the model? Otherwise it will be set to zero.
        """ 

        # add the input parameters
        self.digits = digits
        self.fit_intercept = fit_intercept
        self.y = y
        self.x = x

        # correct the shape of y and x
        Y = np.atleast_2d(y)
        if Y.shape[1] != 1 or Y.ndim < 2:
            Y = Y.T
        X = np.atleast_2d(self.x)
        if X.shape[0] != Y.shape[0] and X.shape[1] == Y.shape[0]:
            X = X.T
        
        # get the matrix of regressors
        if fit_intercept:
            X = np.hstack([np.ones(Y.shape), X])
        
        # get the coefficients and intercept
        self.coefs = np.round(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y), digits)
        if fit_intercept:
            self.intercept = self.coefs[0][0]
            self.coefs = self.coefs[1:].flatten()
        else:
            self.intercept = 0
            self.coefs = self.coefs.flatten()



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
            X = np.atleast_2d(x).T if x.ndim == 1 else x
            return np.sum(X * self.coefs, 1).flatten() + self.intercept



    def __repr__(self):
        return self.__str__()



    def __str__(self):
        t = "Y = " + (str(self.intercept) if self.fit_intercept != 0 else "")
        for i in np.arange(len(self.coefs)):
            t += (" + " if self.coefs[i] >= 0 else " - ") + str(abs(self.coefs[i])) +  " X" + str(i + 1)
        return t



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



class PolynomialRegression(LinearRegression):


    def __init__(self, y, x, order=None, digits=5, fit_intercept=True):
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

            digits:         (int)
                            the number of digits available for each coefficient.

            fit_intercept:  (bool)
                            Should the intercept be included in the model?
        """ 

        # store the entered parameters
        self.y = np.array([y]).flatten()
        self.x = np.array([x]).flatten()
        self.fit_intercept = fit_intercept
        self.digits = digits
        
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
            super(PolynomialRegression, self).__init__(self.y, X, self.digits, self.fit_intercept)

        # fit the model with the given order
        else:
            self.__fit__(order)



    def __fit__(self, order):
        """
        It returns the set of coefficients corresponding to a model with the provided order.
        """
        self.order = order
        X = np.atleast_2d([self.x ** i for i in np.arange(self.order) + 1]).T
        super(PolynomialRegression, self).__init__(self.y, X, self.digits, self.fit_intercept)
        self.x = X[:, 0].flatten()


    
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
        
        # get delta as the offset of y
        self.alpha = np.round(np.min(self.y), digits)
        
        # get Y (ensure no values are zero)
        Y = np.atleast_2d(np.log(y - self.alpha + np.finfo(float).eps)).T
                
        # get beta as the offset of x
        self.gamma = np.round(1 - np.round(np.min(x), digits), digits)
        
        # get X (ensure no values are zero)
        X = np.hstack([np.ones(Y.shape), np.atleast_2d(np.log(x + self.beta + np.finfo(float).eps)).T])
                
        # get the coefficients
        coefs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        
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
        predict use the fitted model to predict regression output according to the model y = alpha * x ** beta.
        """
        return self.alpha + self.beta * (x + self.gamma) ** self.delta



    def __repr__(self):
        return self.__str__()



    def __str__(self):
        return "Y = {} + {} * (X + {}) ** {}".format(self.alpha, self.beta, self.gamma, self.delta)



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