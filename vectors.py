# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:32:16 2018

@author: lzoffoli
"""



class Vector():
    """
    class representing an n-dimensional vector sampled over time.

    Input:
        data: (None or dict)
            the data to be imported as dimensions.
        index: (None, 1d array or list)
            the data DataFrame index. If "None" it will be set as an incremental array.
        xunit: (str)
            the unit of measurement of the x array.
        dunit: (str)
            the unit of measurement of the data provided.
        type: (str)
            the type of data.
    """

    def __init__(self, data=None, index=None, xunit='', dunit='', type='Generic'):

        # import the required packages
        from numpy import sum, diff, any, array, mean
        from pandas import DataFrame

        # check the entered data
        classes = array(['ndarray', 'list', 'NoneType'])
        data = {} if data is None else data
        assert data.__class__.__name__ in ["dict"], '"data" must be a dict object.'
        assert sum(diff([len(data[k]) for k in data])) == 0, '"data" keys must have equal number of samples.' 
        assert ~any([k in ['__'] for k in data]), "'data' keys cannot be any of " + str(classes) + "."

        # check the "index" parameter
        assert any(classes == index.__class__.__name__), "'index' must be any of " + str(classes) + "."
        index = array(index).flatten()

        # check "xunit"
        assert xunit.__class__.__name__ == "str", "'xunit' must be a str object."

        # check "dunit"
        assert dunit.__class__.__name__ == "str", "'dunit' must be a str object."

        # check "type"
        assert type.__class__.__name__ == "str", "'type' must be a str object."

        # data
        setattr(self, "df", DataFrame(data, index=index))

        # add units
        setattr(self, "xunit", xunit)
        setattr(self, "dunit", dunit)

        # add type
        setattr(self, "type", type)

        # add sampling rate
        if self.df.shape[0] > 1 and mean(diff(self.df.index.to_numpy())) != 0:
            setattr(self, "sampling_frequency", 1. / mean(diff(self.df.index)))
        else:
            setattr(self, "sampling_frequency", None)


    def __str__(self):
        out = ["\n\nAttributes:\n\n\ttype:\t", self.type, "\n\txunit:\t", self.xunit, "\n\tdunit:\t",
               self.dunit, "\n"]
        return self.df.__str__() + " ".join(out)


    def __repr__(self):
        return self.__str__()


    def __add__(self, other):
        dt = self.df + (other.df if self.match(other) else other)
        return Vector(dt.to_dict("list"), self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def __sub__(self, other):
        dt = self.df - (other.df if self.match(other) else other)
        return Vector(dt.to_dict("list"), self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)

    def __mul__(self, other):
        dt = self.df * (other.df if self.match(other) else other)
        return Vector(dt.to_dict("list"), self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def __floordiv__(self, other):
        dt = self.df // (other.df if self.match(other) else other)
        return Vector(dt.to_dict("list"), self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def __truediv__(self, other):
        dt = self.df / (other.df if self.match(other) else other)
        return Vector(dt.to_dict("list"), self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def __mod__(self, other):
        dt = self.df % (other.df if self.match(other) else other)
        return Vector(dt.to_dict("list"), self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def __pow__(self, other):
        dt = self.df ** (other.df if self.match(other) else other)
        return Vector(dt.to_dict("list"), self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def __and__(self, other):
        dt = self.df & (other.df if self.match(other) else other)
        return Vector(dt.to_dict("list"), self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def __xor__(self, other):
        dt = self.df ^ (other.df if self.match(other) else other)
        return Vector(dt.to_dict("list"), self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def __or__(self, other):
        dt = self.df | (other.df if self.match(other) else other)
        return Vector(dt.to_dict("list"), self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def __iadd__(self, other):
        self.df += (other.df if self.match(other) else other)


    def __isub__(self, other):
        self.df -= (other.df if self.match(other) else other)


    def __imul__(self, other):
        self.df *= (other.df if self.match(other) else other)


    def __idiv__(self, other):
        self.df /= (other.df if self.match(other) else other)


    def __ifloordiv__(self, other):
        self.df //= (other.df if self.match(other) else other)


    def __imod__(self, other):
        self.df %= (other.df if self.match(other) else other)


    def __ilshift__(self, other):
        self.df <<= (other.df if self.match(other) else other)


    def __irshift__(self, other):
        self.df >>= (other.df if self.match(other) else other)


    def __iand__(self, other):
        self.df &= (other.df if self.match(other) else other)


    def __ixor__(self, other):
        self.df ^= (other.df if self.match(other) else other)


    def __ior__(self, other):
        self.df |= (other.df if self.match(other) else other)


    def __neg__(self):
        return self.df * (-1)


    def __pos__(self):
        return self.df


    def __abs__(self):
        return self.df.abs()


    def __invert__(self):
        return ~self.df


    def __complex__(self):
        return self.df.astype('complex')


    def __int__(self):
        return self.df.astype('int')


    def __long__(self):
        return self.df.astype('long')


    def __float__(self):
        return self.df.astype('float')


    def __oct__(self):
        return self.df.astype('oct')


    def __hex__(self):
        return self.df.astype('')


    def __lt__(self, other):
        return self.df < (other.df if self.match(other) else other)


    def __le__(self, other):
        return self.df <= (other.df if self.match(other) else other)


    def __eq__(self, other):
        return self.df == (other.df if self.match(other) else other)


    def __ne__(self, other):
        return self.df != (other.df if self.match(other) else other)


    def __ge__(self, other):
        return self.df >= (other.df if self.match(other) else other)


    def __gt__(self, other):
        return self.df > (other.df if self.match(other) else other)


    def __getitem__(self, key):
        """
        by default Vector will return the items as if it would be a pandas DataFrame.

        :param key: (str) the item required.
        :return: the item.
        """
        try: obj = self.df[key]
        except Exception: obj = self[key]
        return obj


    def __setitem__(self, key, value):
        """
        by default Vector will work as if it would be a pandas DataFrame.

        :param key: (str) the item required.
        :return: the item.
        """
        try: self.df[key] = value
        except Exception: self[key] = value


    def __getattr__(self, key):
        """
        try to return the attributes from either the main class or DataFrame

        :param key: (str) the name of the attribute required.
        :return: the attribute (if found), None otherwise
        """
        try: obj = vars(self)[key]
        except KeyError: obj = getattr(self.df, key)
        return obj

    '''
    def __setattr__(self, key, value):
        """
        try to set the key attribute to the correct attribute of Vector.

        :param key: the name of the attribute required.
        :param value: the value to be set.
        """
        if isinstance(self.df, key): self.df[key] = value
        else: self[key] = value
    '''

    def copy(self):
        """
        return a copy of the current vector
        """
        return Vector(self.to_dict('list'), self.index.to_numpy(), self.xunit, self.dunit, self.type)


    def diff_winter(self):
        '''
        return the "mean" difference resulting in the differential at each given instant.

        References:
            Winter DA.
                Biomechanics and Motor Control of Human Movement. Fourth Ed.
                Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
        '''

        # import dependancies
        import numpy as np

        # get a copy of the vector
        v = self.subsample(index=self.index.to_numpy()[1:-1])
        
        # get the difference
        v.df.loc[v.index] = (self.values[2:] - self.values[:-2]) * 0.5

        # return the outcome
        return v


    def module(self):
        """
        Get the module of the vector.
        """
        # import the required packages
        from numpy import sqrt, sum

        # get the module as dict
        module = {"|" + " + ".join(self.df.columns) + "|": sqrt(sum(self.df.values ** 2, 1))}

        # return the corresponding Vector
        return Vector(module, self.index.to_numpy(), self.xunit, self.dunit, self.type)


    def normalize(self):
        """
        normalize the whole vector to its module in order to create a unit length vector
        """

        # get the normalization factor
        N = self.module().values.flatten()

        # normalize each dimension
        V = self.copy()
        for d in self.columns:
            V.df.loc[self.index, d] = V.df[d].values.flatten() / N

        # return V
        return V   


    def der1(self, den=None):
        '''
        return the first derivative of the current vector using the Winter (2009)'s approach.

        Input:
            den: (Vector)
                the denominator to be used for the calculatin of the derivative. if None, the time samples are used.

        Output:
            V: (Vector)
                a new vector with shape reduced by 2 samples (first and last) containing the first derivative.
                
        References:
            Winter DA.
                Biomechanics and Motor Control of Human Movement. Fourth Ed.
                Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
        '''

        # import dependancies
        import numpy as np
        from pybiomech.utils import classcheck

        # check dependancies and get the denominator
        if den is not None:
            classcheck(den, ['Vector'])
            assert self.match(den), "den and self must match."
            dd = den.values[1:] - den.values[:-1]
        else:
            dd = self.index.to_numpy()[2:] - self.index.to_numpy()[:-2]
            dd = np.vstack(np.atleast_2d([dd for i in self.columns])).T

        # set the denominator to an extremely high value to avoid division by zero (it sets the limit to zero)
        dd[np.argwhere(dd == 0)] = 2 ** 64 - 1

        # derivate all dimensions
        a = self.subsample(index=self.index.to_numpy()[1:-1])
        dn = self.values[2:] - self.values[:-2]
        a.df.loc[a.index] = dn / dd
        
        # update the unit
        a.dunit = self.dunit + '/' + self.xunit

        # return the data
        return a


    def der2(self):
        '''
        return the first derivative of the current vector using the Winter (2009)'s approach.

        References:
            Winter DA.
                Biomechanics and Motor Control of Human Movement. Fourth Ed.
                Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
        '''

        # import dependancies
        import numpy as np

        # derivate all dimensions twice
        a = self.subsample(index=self.index.to_numpy()[1:-1])
        t = np.vstack(np.atleast_2d([self.index.to_numpy() for i in a.columns])).T
        x = self.values
        dv = (x[2:] - x[1:-1]) / (t[2:] - t[1:-1]) - (x[1:-1] - x[:-2]) / (t[1:-1] - t[:-2])
        dt = (t[2:] - t[:-2]) * 0.5
        a.df.loc[a.index] = dv / dt

        # update the unit
        a.dunit = self.dunit + '/' + self.xunit + "^2"

        # return the data
        return a


    def subsample(self, samples=None, dims=None, index=None):
        """
        Get a subsample of the current vector according to the provided index, samples or dimensions

        Input:
            index: (ndarray)
                a list providing the indices to be retained. If both index and samples are None, all the indices are
                retained

            samples: (ndarray)
                if index is None, it provides the samples to be retained. If index is not None, samples is ignored.

            dims: (ndarray)
                a list with the dimensions (columns) to be retained. if None, all the dimensions are retained.

        Output:
            vec: (Vector)
                a vector which is a subsample of the current vector
        """

        # dependancies
        import numpy as np
        from pybiomech.utils import classcheck

        # check the index
        classcheck(index, ["list", "ndarray", "NoneType"])
        if index is not None:
            idx = np.array([i for i in self.index if i in index])

        # check the samples
        elif samples is not None:
            classcheck(samples, ["list", "ndarray", "NoneType"])
            idx = self.index.to_numpy()[samples]
        
        # keep all samples
        else:
            idx = self.index.to_numpy()

        # check the dimensions
        classcheck(dims, ["list", "ndarray", "NoneType"])
        if dims is None:
            dims = self.columns.to_numpy()
        else:
            dims = np.array([i for i in self.columns if i in dims])

        # return the subsample
        v = self.copy()
        v.df = v.loc[idx, dims]
        return v


    def angle(self, A, B):
        """
        Get the angle between 3 vectors having self as fulcrum. Calculation are made according to the Carnot theorem.

        Input:
            A, B: (Vector)
                Vectors defining the ends of the angle.
        Output:
            a: (float)
                the angle A-self-B in radiants.
        """

        # import the necessary packages
        from numpy import arccos

        # check vdf
        assert self.match(A), '"A" must be a Vector object with index and columns equal to self.'
        assert self.match(B), '"B" must be a Vector object with index and columns equal to self.'

        # get the angle using the Carnot theorem
        a = (A - self).module().values.flatten()
        b = (B - self).module().values.flatten()
        c = (A - B).module().values.flatten()
        angle = arccos((c ** 2 - a ** 2 - b ** 2) / (-2 * a * b))
        return Vector({'Angle': angle}, self.index.to_numpy(), self.xunit, 'rad', 'Angle')


    def rotateby(self, R):
        """
        Rotate the vector by a given rotation matrix or a set of rotation matrices.
        
        Input:
            R: (ndarray, dict)
                if a ndarray is provided, the vector is rotated according to R. Otherwise, the dict must contain a
                rotation matrix for each index in the current vector. Then, the corresponding rotation is adopted for
                each sample.

                The rotation performed is V_rotated = R * V

        Output:
            V_rotated: (Vector)
                a copy of the current vector rotated by R
        """

        # import the necessary packages
        import numpy as np

        # check the entered arguments
        if R.__class__.__name__ == "dict":
            txt = "all keys in R must be indices of the current vector."
            assert np.all([i in self.index.to_numpy() for i in R.keys()]), txt
            for i in R:
                txt = "all R elements must be a square matrix with " + str(self.shape[1]) + " dimensions."
                assert np.all([j == self.shape[1] for j in R[i].shape]), txt
        else:
            assert R.__class__.__name__ == "ndarray", "'R' must be a 'dict' or a 'ndarray'."
            txt = "R must be a square matrix with " + str(self.shape[1]) + " dimensions."
            assert np.all([j == self.shape[1] for j in R.shape]), txt
        
        # prepare R to be used
        if R.__class__.__name__ != "dict":
            R = {i: R for i in self.index}
        
        # rotate each sample in self
        V = self.copy()
        for i in R:
            for j, v in zip(self.columns.to_numpy(), R[i].dot(np.atleast_2d(self.loc[i].values).T)):
                V.df.loc[i, j] = v

        # return the rotate Vector
        return V


    def interpolate(self, n=None, x_old=None, x_new=None):
        """
        cubic spline interpolate the current vector.

        Input:
            n : (int)
                a given number of samples. In this scenario, the index will be replaced by range(n).

            x_old: (list, ndarray)
                if provided, it represents the reference to be used for the interpolation of the data in self.
                it must be 1D array with length equal to the number of samples of self.

            x_new : (list, ndarray)
                a 1D array or a list that can be casted to a 1D array that represents the new index
                of the interpolated data.

        Output:
            V: (Vector)
                a vector equal to self by with samples defined as described above.
        """

        # dependancies
        import pybiomech.processing as pr
        from pybiomech.utils import classcheck
        import numpy as np

        # check the entered data
        classcheck(n, ['int', 'NoneType'])
        classcheck(x_new, ['list', 'ndarray', 'NoneType'])
        classcheck(x_old, ['list', 'ndarray', 'NoneType'])
        if x_old is not None:
            x_old = np.squeeze(np.array([x_old]))
            assert x_old.ndim == 1, "'x_old' must have only 1 dimension."
            assert len(x_old) == self.shape[0], "'x_old' length must be " + str(self.shape[0]) + "."
        else:
            x_old = self.index.to_numpy().flatten()
        assert n is not None or x_new is not None, "'x_new' or 'n' must be provided."
        assert n is None or x_new is None, "One between 'x_new' or 'n' must be None."
        if n is not None:
            assert n > 0, "'n' must be an int > 0."
            idx = np.arange(n)
        if x_new is not None:
            x_new = np.array([x_new]).flatten()
            idx = x_new

        # interpolate the data
        v = {i: pr.interpolate(self[i].values.flatten(), n, x_old, x_new)[0] for i in self.columns}
        return Vector(v, idx, self.xunit, self.dunit, self.type)


    def match(self, vector, verbose=False):
        """
        Test the equality between self and vector.

        Input:
            vector:
                an object
            verbose: (bool)
                if True feedback about the possible differences between self and vector are provided.

        Output:
            True if self has same dimensions, index, type, xunit and dunit. False otherwise.
        """

        # check vector
        cl = vector.__class__.__name__ == "Vector"
        if not cl and verbose: print("'vector' must be an object of class 'Vector'.")

        # check the index
        try: i = all(self.index.to_numpy() - vector.index.to_numpy() == 0)
        except Exception: i = False
        if not i and verbose: print("'index' differs between 'self' and 'vector'.")

        # check the xunit
        try: x = self.xunit == vector.xunit
        except Exception: x = False
        if not x and verbose: print("'xunit' differs between 'self' and 'vector'.")

        # check the dunit
        try: d = self.dunit == vector.dunit
        except Exception: d = False
        if not d and verbose: print("'dunit' differs between 'self' and 'vector'.")

        # check the type
        # try: t = self.type == vector.type
        # except Exception: t = False
        # if not t and verbose: print("'type' differs between 'self' and 'vector'.")

        # check the dimensions
        try: c = all([self.columns[i] == vector.columns[i] for i in range(vector.shape[1])])
        except Exception: c = False
        if not c and verbose: print("'dimensions' differs between 'self' and 'vector'.")

        # return the outcome
        return cl & i & x & d & c # & t


    def nanreplace(self, vectors, C=None, scoring="r2", plot=None, max_tr_data=1000, *args, **kwargs):
        """
        Use Support Vector Regression (SVR) to provide the coordinates of the missing samples in the current vector.
        Where this is not possible, cubic spline interpolation is employed.

        References:
            Smola A. J., Schölkopf B. (2004).
            A tutorial on support vector regression. Statistics and Computing, 14(3), 	199–222.

        Input:
            vectors: (Vector)
                Other vectors that can be used as reference for the algorithm to let it learn where the current
                vector should be in space and time.
            C: (None, list, 1D array)
                The penalty to be applied to errors in the calculation of SVR.
                If a list or an array is passed, the C value resulting in the higher SVR score will be used.
                If None a several C values will be tested. C must be strictly positive.
            plot: (None or str)
                If None, no plots are generated. If plot is the path to an html or png file, the plots showing the
                results of the nan replacement are generated.
            scoring: (str)
                a string or an object created via the "make_score" method available from the sklearn package.
            max_tr_data: (int)
                the maximum number of training data to be used. If the effective available number is lower than it,
                all the training data are used. Otherwise, the specified numebr is randomly sampled from the
                available pool.
            args:
                parameters passed to the SVR function.
            kwargs:
                other parameters passed to the SVR algorithm.

        Output:
            new_vector: (Vector)
                a new Vector equal to the current one bust without missing data.
            Summary: (pandas DataFrame)
                A summary table that shows some data about the procedure.
            plot: (optional)
                if plot is not None, one plot is generated and stored in the provided path.
        """

        # import the necessary packages
        import warnings
        import numpy as np
        import matplotlib.pyplot as pl
        import matplotlib.gridspec as gs
        import pybiomech.plot as pp
        from pybiomech.processing import interpolate
        from sklearn.svm import SVR
        from sklearn.model_selection import GridSearchCV
        from sklearn.exceptions import ConvergenceWarning
        from pandas import DataFrame
        from os import makedirs
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # get a copy of the current vector
        vcopy = self.copy()

        # generate the summary table
        fit = DataFrame()

        # check vectors
        txt = "'vectors' must contain only 'Vector' objects."
        assert all([vectors[i].__class__.__name__ == "Vector" for i in vectors]), txt
        assert all([vectors[i].match(self) for i in vectors]), "'Support vectors must match self."

        # check C
        if C is None: C = np.unique([j * (10 ** k) for j in np.arange(1, 11) for k in np.linspace(-15, 1, 17)])
        C = np.array(C).flatten()
        assert np.all(C > 0), "'C' must be positive values."
        C = np.array(C).flatten().astype(float)

        # get the missing values
        y = self.values
        x = np.hstack([vectors[v].values for v in vectors])
        miss_idx = np.argwhere(np.any(np.isnan(y), 1).flatten()).flatten()
        train_idx = np.argwhere((~np.any(np.isnan(y), 1)) & (~np.any(np.isnan(x), 1))).flatten()
        still_miss = np.argwhere((np.any(np.isnan(x), 1)) & (np.any(np.isnan(y), 1))).flatten()

        # remove those columns resulting in nan values along the still_miss index
        if len(still_miss) > 0:
            x = x[:, ~np.any(np.isnan(x[still_miss, :]), axis=0)]
            txt = "Relationship vectors were not able to extract data having indices:\n"
            txt += str(self.index[still_miss])
            if x.shape[1] <= y.shape[1]:

                # The removal of non-necessary relationships is not possible, thus use cubic spline interpolation to
                # fit the missing points in x
                x = np.hstack([vectors[v].values for v in vectors])
                for i, v in enumerate(x.T):
                    missing = still_miss[np.argwhere(np.isnan(v[still_miss])).flatten()]
                    if len(missing) > 0:
                        valid = np.argwhere(~np.isnan(v)).flatten()
                        idx = self.index.to_numpy()
                        interpolated = interpolate(v[valid], len(v), idx[valid], idx)[0]
                        x[missing, i] = interpolated[missing]

        # grid searcher
        grid = GridSearchCV(estimator=SVR(*args, **kwargs), param_grid={'C': C}, scoring=scoring, cv=5, iid=False)

        # work on each vector
        for i, v in enumerate(self.df.columns):
            if len(miss_idx) > 0:

                # minimize training data
                tr_data = np.concatenate((np.atleast_2d(vcopy[v].values).T, x), axis=1)[train_idx]
                tr_data = tr_data[np.argwhere(~np.any(np.isnan(tr_data), 1)).flatten(), :]
                tr_test = tr_data[np.unique(tr_data, axis=0, return_index=True)[1], :]

                # select max_tr_data samples at random
                np.random.seed()
                tr_data = tr_test[np.random.permutation(np.arange(tr_test.shape[0]))[:max_tr_data], :]

                # get the best estimator
                est = grid.fit(tr_data[:, 1:], tr_data[:, 0])

                # find missing points
                vcopy.loc[vcopy.index[miss_idx], v] = est.best_estimator_.predict(x[miss_idx, :])

                # generate the SVR summary table for the current axis
                rmse = tr_data[:, 0].flatten() - est.best_estimator_.predict(tr_data[:, 1:]).flatten()
                rmse = np.sqrt(np.mean(rmse ** 2))
                tab = {'Dimension': [v], 'missing_n': [len(miss_idx)],
                       'missing_%': [100. * len(miss_idx) / y.shape[0]],
                       'C': [est.best_params_['C']], 'Est_RMSE': [rmse]}

            # SVR was not necessary, thus generate an empty summary table
            else: tab = {'Dimension': [v], 'missing_n': [len(miss_idx)],
                         'missing_%': [100. * len(miss_idx) / y.shape[0]], 'C': [None], 'Est_RMSE': [None]}

            # update the output table
            fit = fit.append(DataFrame(tab), sort=False)

        # update the fit index
        fit.index = np.arange(fit.shape[0])

        # plot data if required
        if plot is not None:

            # check data
            assert plot[-3:] == "png", "'plot' must be a valid png file."

            # ensure all the folders have been created
            makedirs("\\".join(plot.split("\\")[:-1]), exist_ok=True)

            # set the output file
            fig = pl.figure(figsize=(pp.cm2in(20), pp.cm2in(20)), dpi=1000, frameon=False)
            grid = gs.GridSpec(self.shape[1], 2)
            plts = [fig.add_subplot(grid[i, :]) for i in np.arange(vcopy.shape[1])]
            for i, v in enumerate(vcopy.columns):

                # create the plots
                x_tr = vcopy.index
                y_tr = vcopy[v].values.flatten() - vcopy[v].min()
                plts[i].scatter(x_tr, y_tr, color='b', label="Original", s=1)
                if len(miss_idx) > 0:
                    x_ms = vcopy.index[miss_idx]
                    y_ms = vcopy.loc[vcopy.index[miss_idx], v] - vcopy[v].min()
                    plts[i].scatter(x_ms, y_ms, color='r', label="Recovered", s=1)
                if i == 0:
                    h, l = plts[i].get_legend_handles_labels()
                    pl.figlegend(h, l, loc="lower center", ncol=vcopy.shape[1], framealpha=0.6,
                                 facecolor="lightyellow", edgecolor='k')

                # set the layout
                x_lab = "" if i < vcopy.shape[1] - 1 else "Time (s)"
                x_ticks_lab = ["" for i in np.arange(5)] if i < vcopy.shape[1] - 1 else None
                pp.set_layout(plts[i], x_label=x_lab, y_label_left=self.dunit + "\n", y_label_right=v + "\n",
                              x_ticks_lab=x_ticks_lab, x_digits="{:0.1f}", y_digits="{:0.2f}")

            # export the output
            pl.suptitle("SVR missing data recovery", x=0.1, va="top", ha="left", fontsize=12, fontweight="bold")
            pl.tight_layout(rect=[0, 0.05, 1, 0.95])
            pl.savefig(plot, transparent=True)
            pl.close()

        # return the updated vector
        return vcopy, fit


    def optimal_cutoffs(self, power=[95, 99], r_n=500, r_ord=4, r_seg=2, r_min=5, plot=None, par=None):
        """
        Calculate the optimal cutoff as the frequencies corresponding to the specified cumulative signal power of
        power of each dimension plus using the Winter's residuals analysis approach.

        Input:
            power: (list)
                a list containing the power(s) level required. The power level must be a positive value in the (0,
                100] range.
            r_n: (int)
                the number of frequencies to be tested within the (0, fs/2) range to create the residuals curve of
                the Winter's residuals analysis approach.
            r_ord: (int)
                the filter for the phase-corrected Butterworth filter used to calculate the optimal cut-off according
                to the Winter's residuals analysis approach.
            r_seg: (int)
                the number of segments that can be used to fit the residuals curve in order to identify the best
                deflection point. It should be a value between 2 and 4.
            r_min: (int)
                the minimum number of elements that have to be considered for each segment during the calculation of
                the best deflection point.
            plot: (None or str)
                if an html or png file path is provided, it is used to generate a plot reflecting the outcome of this
                method.
            par: (None or Pool)
                a parallel object that is passed to the res_an function.

        Output:
            a pandas DataFrame where each row will be a dimension and each column will contain the cutoff
            corresponding to the cumulative signal power and to the Winter's residuals analysis approach.
        """

        # import dependancies
        from pybiomech.processing import power_freq, res_an, power_spectrum
        from pandas import DataFrame, Series
        import matplotlib.pyplot as pl
        import matplotlib.gridspec as gs
        import pybiomech.plot as pp
        import numpy as np

        # check the entered parameters
        assert power.__class__.__name__ == "list", "'power' must be a list."
        txt = "All elements of power must be int or float."
        assert all([i.__class__.__name__ in ["int", "float"] for i in power]), txt
        assert r_n.__class__.__name__ == "int", "'res_an_n' must be an int."
        assert r_n > 0, "'res_an_n' must be > 0."
        assert r_ord.__class__.__name__ == "int", "'res_an_ord' must be an int."
        assert r_ord > 0, "'res_an_ord' must be > 0."
        assert r_seg.__class__.__name__ == "int", "'res_an_seg' must be an int."
        assert r_seg >= 2, "'res_an_seg' must be >= 2."
        assert r_min.__class__.__name__ == "int", "'res_an_min' must be an int."
        assert r_min >= 2, "'res_an_min' must be >= 2."
        if plot is not None: assert plot.split(".")[-1] in ["png"], "'plot' must be a 'png' file."
        assert par.__class__.__name__ in ["NoneType", "Pool"], "'par' must be None or a Pool object."

        # get the output dataframe
        df = DataFrame()

        # generate the figure and the colormap if necessary
        if plot is not None:
            handles = []
            labels = []
            fig = pl.figure(figsize=(pp.cm2in(6 * (self.shape[1] + 1)), pp.cm2in(20)), dpi=1000, frameon=False)
            grid = gs.GridSpec(3, self.shape[1])
            plts = [[fig.add_subplot(grid[i, j]) for j in np.arange(self.shape[1])] for i in np.arange(3)]
            colors = pp.get_colors_from_map(len(power) + 5, 'brg')


        # work on each dimension
        for i, v in enumerate(self.columns):

            # generate the output line
            line = DataFrame()

            # get the signal power
            for j, p in enumerate(power):
                popt, pfrq, pwr = power_freq(self.df[v].values, p, self.sampling_frequency)
                line[str(p) + "% cumulative power (Hz)"] = Series(popt)

                # generate the cumulative signal power plot if required
                if plot is not None:
                    x = pfrq[pfrq <= 5 *popt]
                    y = pwr[pfrq <= 5 * popt]
                    if j == 0: plts[0][i].plot(pfrq, pwr, label=r"$\sum(Power)$", color=colors[0])
                    idx = np.argwhere(x >= popt).flatten()[0]
                    plts[0][i].scatter(x[idx], y[idx], color=colors[len(colors) - 1 - j],
                                       label=str(p) + "% " + r"$\sum{Power}$")
                    txt = r"$\sum{Power}=%0.2f$" % pwr[idx]
                    txt += " %\n"
                    txt += "$Frequency=%0.2f Hz$" % pfrq[idx]
                    corr = 1. / (len(power) + 1) * (j + 1)
                    plts[0][i].annotate(txt, xy=(x[idx], y[idx]), xycoords='data', ha='left', va='top',
                                        xytext=(np.max(pfrq) * corr, np.max(pwr) * corr),
                                        textcoords='data', arrowprops={'arrowstyle': "->"}, fontsize=6)

            # setup the layout if plot is required
            if plot is not None:
                x_loc = np.linspace(0, self.sampling_frequency * 0.5, 5)
                x_lab = ["%0.0f" % i for i in x_loc]
                y_ticks_loc = np.round(np.linspace(0, 100, 5), 0)
                y_lab = "" if i > 0 else ("%")
                pp.set_layout(plts[0][i], y_label_left=y_lab, title=v, label_size=8, x_ticks_text_size=8,
                              y_ticks_text_size=8, y_ticks_loc=y_ticks_loc, x_ticks_loc=x_loc,
                              x_ticks_lab=x_lab, y_digits="{:0.0f}")

            # get the residual analysis cutoff
            ropt, rfrq, res = res_an(self.df[v].values, self.sampling_frequency, r_n, r_ord, r_seg, r_min, par)[:3]
            line['Residual analysis (Hz)'] = Series(ropt)

            # plot the residual analysis outcomes
            if plot is not None:

                # plot the data
                z = res[rfrq >= ropt][0]
                x = rfrq
                y = np.array([res]).flatten()
                mag = int(np.floor(np.log(np.max(y)) / np.log(10))) if np.max(res) > 0 else 0
                if not -1 < mag < 1:
                    y /= (10 ** mag)
                    z /= (10 ** mag)
                plts[1][i].plot(x, y, label="Residuals", color=colors[1])
                plts[1][i].scatter(ropt, z, label="Optimal cutoff", color=colors[len(colors) - len(power) - 1])
                mul = "" if -1 < mag < 1 else ("$x 10^{" + str(mag) + "}$")
                txt = "$Residuals=%0.2f$" % z
                txt += mul + self.dunit + "$^{2}$" + "\n"
                txt += "$Frequency=%0.2f Hz$" % ropt
                plts[1][i].annotate(txt, xy=(ropt, z), xycoords='data', ha='left', va='top', fontsize=6,
                                    xytext=(np.max(x) * 0.3, np.max(y) * 0.7), textcoords='data',
                                    arrowprops={'arrowstyle': "->"})

                # set the layout
                plts[1][i].text(0, np.max(y) * 1.05, mul, ha="right", va="bottom", fontsize=6)
                y_lab = "" if i > 0 else (self.dunit + "$^{2}$")
                if np.all(np.diff(res) == 0):
                    y_ticks_loc = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]) + np.mean(res)
                else:
                    y_ticks_loc = None
                pp.set_layout(plts[1][i], y_label_left=y_lab, label_size=8, x_ticks_text_size=8, y_ticks_text_size=8,
                              x_ticks_loc=x_loc, x_ticks_lab=x_lab, y_ticks_loc=y_ticks_loc)

            # get the Power Spectrum of the signal
            sfrq, spe = power_spectrum(self.df[v].values - np.mean(self.df[v].values), self.sampling_frequency)

            # get the frequency corresponding to the peak power
            pkf = sfrq[np.argmax(spe[1:]) + 1]
            pkp = np.max(spe[1:])
            line['Peak power frequency (Hz)'] = Series(pkf)

            # plot the power spectrum if required
            if plot is not None:

                # plot the data
                rng = np.argwhere(sfrq <= 5).flatten()
                x = sfrq[rng]
                y = np.array([spe]).flatten()[rng]
                xp = pkf
                yp = pkp
                psum = np.sum(spe)
                mag = int(np.floor(np.log(pkp) / np.log(10))) if pkp > 0 else 0
                if not -1 < mag < 2:
                    y /= (10 ** mag)
                    yp /= (10 ** mag)
                    psum /= (10 ** mag)
                xa = np.max(x) * 0.4
                ya = yp * 0.5
                plts[2][i].plot(x, y, label="Power Spectrum", color=colors[2])
                plts[2][i].scatter(xp, yp, label="Peak Power", color=colors[len(colors) - len(power) - 2])
                mul = "" if -1 < mag < 1 else ("$x 10^{" + str(mag) + "}$")
                plts[2][i].text(0, yp * 1.05, mul, ha="right", va="bottom", fontsize=6)
                txt = "$Frequency=%0.3f Hz$" % pkf
                plts[2][i].annotate(txt, xy=(xp, yp), xycoords='data', ha='left', va='top', fontsize=6,
                                    xytext=(xa, ya), textcoords='data', arrowprops={'arrowstyle': "->"})

                # set the layout
                pp.set_layout(plts[2][i], y_label_left=self.dunit + "$^{2}$" if i == 0 else "", y_digits="{:0.2f}",
                              x_label="Frequency (Hz)", label_size=8, x_ticks_text_size=8, y_ticks_text_size=8)

            # store the line
            df = df.append(line, sort=False)

        # finalize the plot if required
        if plot is not None:

            # plot the title
            pl.suptitle("Frequency domain analysis", x=0.1, va="top", ha="left", fontsize=12, fontweight="bold")

            # plot the legend
            for i in plts:
                h, l = i[0].get_legend_handles_labels()
                handles += h
                labels += l
            pl.figlegend(handles, labels, 'center right', ncol=1, framealpha=0.6, edgecolor='k',
                         facecolor="lightyellow", fontsize=6)

            # adjust the layout
            try: pl.tight_layout(rect=[0, 0, 0.88, 0.95])
            except Exception: pass
            pl.savefig(plot, transparent=True)
            pl.close()

        # add vector's information to the dataframe
        df.insert(0, "Dimension", [str(i) for i in self.df.columns])
        df.insert(0, "Type", [str(self.type) for i in self.df.columns])
        df.index = np.arange(df.shape[0])

        # return the output
        return df


    def residual_analysis_cutoffs(self, r_n=None, r_ord=4, r_seg=2, r_min=2, plot=None):
        """
        Calculate the optimal cutoff via a modified Winter (2009) approach.

        Input:
            r_n: (None, int)
                the number of frequencies to be tested within the (0, fs/4) range to create the residuals curve of
                the Winter's residuals analysis approach. If None, a 0.5 Hz window is created to cover all the available
                freuqnecy spectrum.
            r_ord: (int)
                the filter for the phase-corrected Butterworth filter used to calculate the optimal cut-off according
                to the Winter's residuals analysis approach.
            r_seg: (int)
                the number of segments that can be used to fit the residuals curve in order to identify the best
                deflection point. It should be a value between 2 and 4.
            r_min: (int)
                the minimum number of elements that have to be considered for each segment during the calculation of
                the best deflection point.
            plot: (None or str)
                if an html or png file path is provided, it is used to generate a plot reflecting the outcome of this
                method.

        Output:
            a pandas DataFrame where each row will be a dimension and each column will contain the cutoff
            corresponding to the cumulative signal power and to a modified Winter's residuals analysis approach.
        """

        # import dependancies
        import pybiomech.processing as pr
        from pandas import DataFrame, Series
        import matplotlib.pyplot as pl
        import matplotlib.gridspec as gs
        import pybiomech.plot as pp
        import pybiomech.utils as ut
        import numpy as np

        # check the entered parameters
        assert r_n.__class__.__name__ in ["NoneType", "int"], "'res_an_n' must be an int."
        assert r_ord.__class__.__name__ == "int", "'res_an_ord' must be an int."
        assert r_ord > 0, "'r_ord' must be > 0."
        assert r_seg.__class__.__name__ == "int", "'res_an_seg' must be an int."
        assert r_seg >= 2, "'r_seg' must be >= 2."
        assert r_min.__class__.__name__ == "int", "'res_an_min' must be an int."
        assert r_min >= 2, "'r_min' must be >= 2."
        if plot is not None: assert plot.split(".")[-1] in ["png"], "'plot' must be a 'png' file."

        # get the output dataframe
        df = DataFrame()

        # get the frequencies to be tested
        fs = self.sampling_frequency
        if r_n is None:

            # since no reference has been provided, the sensitivity is adapted according to the available frequency
            # span. To this purpose, a sensitivity threshold is provided (T = 3) and it is subtracted to the
            # magnitude of the upper limit of the frequency spectrum (U). Their difference will be the
            # output sensitivity (D)

            # get the upper limit of the spectrum
            K = fs * 0.25

            # get its magnitude
            U = ut.magnitude(K)

            # set the sensitivity threshold (the lower the higher the sensitivity)
            T = 3

            # get the order of sensitivity by subtracting T from the magnitude of U
            D = 10 ** (U - T)

            # get X
            x = np.arange(D, K, D)

        # r_n is an int so separate the frequency spectrum in r_n samples
        else:
            x = np.linspace(0, fs * 0.25, r_n + 2)[1:-1]

        # generate the figure and the colormap if necessary
        if plot is not None:
            fig = pl.figure(figsize=(pp.cm2in(6 * (self.shape[1])), pp.cm2in(6)), dpi=1000, frameon=False)
            grid = gs.GridSpec(1, self.shape[1])
            plts = [fig.add_subplot(grid[i]) for i in np.arange(self.shape[1])]

        # work on each dimension
        for i, v in enumerate(self.columns):

            # generate the output line
            line = DataFrame()

            # get the residuals
            y = self.df[v].values
            if np.all(np.float32(y - np.mean(y)) == 0):

                # set y
                y = np.tile(0, len(x))

                # set the crossover at 1
                cp = 1
            else:

                # get y
                y = np.array([np.sum((y - pr.filt(y, r_ord, j, fs)) ** 2) for j in x])
                y /= np.max(y)
                y *= 100

                # get the optimal crossover
                cp = pr.crossovers(y, K=2, samples=2)[0][0]
                cp += pr.crossovers(y[cp:], K=2, samples=2)[0][0]

            # get the fitting line
            fit = np.polyfit(x[cp:], y[cp:], 1)
            ln = np.polyval(fit, x)

            # get the Winter's cut-off
            it = np.tile(fit[-1], len(x))
            wt_x = x[np.argwhere(y >= fit[-1]).flatten()[-1]]
            wt_y = y[np.argwhere(y >= fit[-1]).flatten()[-1]]
            wt_txt = r"$Cut\ off=" + "{:0.1f}".format(wt_x) + "\ Hz$"
            line['Cut off (Hz)'] = Series(wt_x)

            # store the line
            df = df.append(line, sort=False)

            # plot the residual analysis outcomes
            if plot is not None:

                # to optimize the plot readability plot data up to 2 times the cut-off
                x_lim = x <= 5 * wt_x

                # adjust the plot in the y axis accordingly
                y_lim = (y <= np.max(x[x_lim]) / (fs * 0.5) * 100)

                # set the proper limits
                lim = np.argwhere(x_lim & y_lim).flatten()
                if len(lim) == 0:
                    lim = np.argwhere(x_lim).flatten()

                # plot the residuals
                plts[i].plot(x[lim], y[lim], color="navy", alpha=0.8, linewidth=0.8)

                # plot the fitting line
                plts[i].plot(x[lim], ln[lim], color="k", alpha=0.5, linewidth=0.6, linestyle="dashed")

                # plot the intersecting line
                plts[i].plot(x[lim], it[lim], color="darkgreen", alpha=0.5, linewidth=0.6, linestyle="dashed")

                # plot the Winter's cut-off
                plts[i].plot(wt_x, wt_y, color="darkred", ms=2, marker="o")
                plts[i].annotate(wt_txt, xy=(wt_x, wt_y), xycoords='data', ha='center', va='bottom',
                                 xytext=(np.mean(x[lim]), np.max(y[lim]) * 0.3), textcoords='data',
                                 arrowprops={'arrowstyle': "->"}, fontsize=6)

                # set the layout
                ylb = "" if i > 0 else r"$Residuals\ (\%)$"
                pp.set_layout(plts[i], r"$Frequency\ (Hz)$", y_label_left=ylb, label_size=6, x_ticks_text_size=6,
                              y_ticks_text_size=6, title=v)

        # finalize the plot if required
        if plot is not None:

            # plot the title
            pl.suptitle("Residuals analysis", x=0.1, va="top", ha="left", fontsize=8, fontweight="bold")

            # adjust the layout
            try: pl.tight_layout(rect=[0, 0, 1, 0.93])
            except Exception: pass
            pl.savefig(plot, transparent=True)
            pl.close()

        # add vector's information to the dataframe
        df.insert(0, "Dimension", [str(i) for i in self.df.columns])
        df.insert(0, "Type", [str(self.type) for i in self.df.columns])
        df.index = np.arange(df.shape[0])

        # return the output
        return df


    def butt_filt(self, cutoff, order=4, type="lowpass", phase_corrected=True, plot_path=None):
        """
        Filter all data in the vector using a Butterworth filter with the provided features.

        Input:
            cutoff: (float, list, array)
                    the filter cut-off in Hz.

            order: (int)
                    the filter order.

            type: ("low", "high", "bandpass", "stopband")
                    any of the defined strings above defining the type of filter.

            phase_corrected: (bool)
                    Should the filter being applied twice in order to avoid alterations of the signals' phase?

            plot_path: (None, str)
                    if a string is provided, a figure is generated in the plot_path showing the results of the filtering procedure.

            freq_lim: (float)
                    the upper frequency limit to be used when the data are plotted in the frequency domain.

        Output:
            O: (Vector)
                a new vector which is a copy of the current one but with filtered signals in each dimension.
        """

        # import dependancies
        from pybiomech.processing import filt, psd
        import matplotlib.pyplot as pl
        import matplotlib.gridspec as gs
        import pybiomech.plot as pp
        import pybiomech.utils as ut
        import numpy as np

        # check the entered parameters
        types = ["low", "lowpass", "high", "highpass", "bandpass", "stopband"]
        assert type in types, "'type' must be any of: " + str(types) + "."
        if type in ["low", "lowpass", "high", "highpass"]:
            ut.classcheck(cutoff, ["float", "int"])
        else:
            assert len(cutoff) == 2, "'cutoff' must be a 2 elements list or ndarray."
            [ut.classcheck(i, ["float", "int"]) for i in cutoff]
        ut.classcheck(order, ["int"])
        assert phase_corrected or not phase_corrected, "'phase_corrected' must be a 'boolean' object."
        ut.classcheck(plot_path, ["NoneType", "str"])
        if plot_path is not None:
            assert plot_path.split(".")[-1] in ["png"], "'plot_path' must be a 'png' file."

        # generate the figure if necessary
        if plot_path is not None:
            fig = pl.figure(figsize=(pp.cm2in(6 * (self.shape[1])), pp.cm2in(12)), dpi=1000, frameon=False)
            grid = gs.GridSpec(2, self.shape[1])
            plts = [[fig.add_subplot(grid[0, i]), fig.add_subplot(grid[1, i])] for i in np.arange(self.shape[1])]

        # create a copy of the current vector
        O = self.copy()

        # work on each dimension
        fs = self.sampling_frequency
        x = self.index.to_numpy()
        for i, v in enumerate(O.columns):

            # filter the data
            O.df.loc[x, v] = filt(self[v].values.flatten(), order, cutoff, self.sampling_frequency, type,
                                  phase_corrected)

            # plot the filtering outcomes
            if plot_path is not None:

                # time-domain plot
                y_old = self[v].values.flatten()
                y_new = O[v].values.flatten()
                plts[i][0].plot(x, y_old, color="navy", alpha=0.5, linewidth=0.6, label="Raw data")
                plts[i][0].plot(x, y_new, color="darkred", alpha=0.5, linewidth=0.6, label="Filtered data")

                # set the layout
                mag = ut.magnitude(np.max(y_old))
                y_dig = "{:0." + str(0 if mag > 0 else abs(mag) + 1) + "f}"
                pp.set_layout(plts[i][0], self.xunit, "" if i > 0 else self.dunit, label_size=6, x_ticks_text_size=6,
                              y_ticks_text_size=6, title=v, y_digits=y_dig)

                # plot the legend
                if i == len(O.columns) - 1:
                    [handles, labels] = plts[i][0].get_legend_handles_labels()
                    plts[i][0].legend(handles, labels, loc='upper right', ncol=1, framealpha=0.4, edgecolor='k',
                                      facecolor="lightyellow", fontsize=6)
            
                # frequency domain plot
                z_old, k_old = psd(self[v].values.flatten() - np.mean(self[v].values.flatten()), fs)
                z_new, k_new = psd(O[v].values.flatten() - np.mean(self[v].values.flatten()), fs)
                n = np.argwhere(k_old <= 2* cutoff).flatten()
                plts[i][1].plot(k_old[n], z_old[n], color="navy", marker="o", markersize=1, alpha=0.5, linewidth=0.4)
                plts[i][1].plot(k_new[n], z_new[n], color="darkred", marker="o", markersize=0.6, alpha=0.5,
                                linewidth=0.4)

                # set the layout
                mag = ut.magnitude(np.max(z_old))
                y_dig = "{:0." + str(0 if mag > 0 else abs(mag) + 1) + "f}"
                pp.set_layout(plts[i][1], "Hz", "" if i > 0 else "$(" + self.dunit + ")^2$", label_size=6,
                              x_ticks_text_size=6, y_ticks_text_size=6, title=v, y_digits=y_dig)

        # finalize the plot if required
        if plot_path is not None:

            # plot the title
            title = "Butterworth filtering\n"
            title += r"$type:\ " + type + ("pass" if type in ["low", "high"] else "") + "$\n"
            title += r"$order:\ "
            if str(order)[-1] == 1:
                title += "{:0.0f}".format(order) + "^{st}$\n"
            elif str(order)[-1] == 2:
                title += "{:0.0f}".format(order) + "^{nd}$\n"
            elif str(order)[-1] == 3:
                title += "{:0.0f}".format(order) + "^{rd}$\n"
            else:
                title += "{:0.0f}".format(order) + "^{th}$\n"
            title += r"$cut\ off:\ {:0.1f}\ Hz".format(cutoff) + "$\n"
            title += r"$phase\ correction:\ " + ("Yes" if phase_corrected else "No") + "$"
            pl.suptitle(title, x=0.1, va="top", ha="left", fontsize=8, fontweight="bold")

            # adjust the layout
            try: pl.tight_layout(rect=[0, 0, 1, 0.80])
            except Exception: pass
            pl.savefig(plot_path, transparent=True)
            pl.close()

        # return the output
        return O


    def to_df(self, wide=False):
        """
        Return a pandas dataframe containing the data of the vector.

        Input:
            wide: (bool)
                If False, the dataframe will be in "long format". The index and dimensions are stored as columns
                with names "Index_ZZZ" (for the index) and the "YYY_ZZZ" for the columns where "YYY" indicates the
                dimension of the vector and "ZZZ" the unit of measurement.
                If True, the dataframe will be created in wide format. This means that each row will be one dimension
                of the vector. The first column will name the dimension, the second the unit of measurement and the
                others the values. These values will have column name in the form "KKK ZZZ" where "KKK" is one index
                value and "ZZZ" the index unit of measurement.

        Output:
            df: (pandas.DataFrame)
                the dataframe containing the data of the vector.
        """

        # check the entered data
        assert wide or not wide, "'wide' must be a bool."

        # get the df
        df = self.df

        # manage wide format
        if wide:

            # transpose the dataframe
            df = df.T

            # update the columns
            df.columns = [str(i) + "_" + self.xunit for i in df.columns]

            # add the column with the remaining information
            df.insert(0, 'Unit', [self.dunit for i in df.index.to_numpy()])
            df.insert(0, 'Dimension', df.index.to_numpy())

        # manage long format
        else:

            # update the columns
            df.columns = [str(i) + "_" + self.dunit for i in df.columns]

            # insert the index column
            df.insert(0, "Index_" + self.xunit, df.index)

        # refresh the index
        df.index = range(df.shape[0])

        # return the index
        return df


    def append(self, other, sort=False, *args, **kwargs):
        """
        wrapper of the pandas.DataFrame append function
        """
        from numpy import append
        dt = self.df.append(other.df if self.match(other) else other, sort=sort, *args, **kwargs)
        idx = append(self.index.to_numpy(), other.index.to_numpy())
        return Vector(dt.to_dict("list"), idx, self.xunit, self.dunit, self.type)


    def dot(self, other):
        """
        wrapper of the pandas.DataFrame dot function
        """
        dt = self.df.dot((other.df if self.match(other) else other).T).T
        dt.columns = self.columns
        return Vector(dt.to_dict("list"), dt.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def cross(self, other, axis=1):
        """
        wrapper of the numpy cross function
        """
        from numpy import cross
        dt = cross(self.df.values, (other.df if self.match(other) else other).values, axis=axis)
        cols = self.columns.to_numpy()
        return Vector({i: v for i, v in zip(cols, dt.T)}, self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def inv(self):
        """
        wrapper of the numpy inv function
        """
        from numpy.linalg import inv
        dt = inv(self.df.values)
        cols = self.columns.to_numpy()
        dt = {i[0]: i[1] for i in zip(cols, dt)}
        return Vector(dt, self.index.to_numpy().flatten(), self.xunit, self.dunit, self.type)


    def mean(self):
        """
        get the mean value of each dimension
        """
        import numpy as np
        dt = {i[0]: [i[1]] for i in zip(self.columns.to_numpy(), self.df.mean(axis=0).values)}
        return Vector(dt, np.array([np.mean(self.index.to_numpy())]), self.xunit, self.dunit, self.type)



class VectorDict(dict):
    """
    Create a dict of "Vector" object(s). It is a simple wrapper of the "dict" class object with additional methods.

    Input:
        args: (objects)
            objects of class vectors.
    """


    def __init__(self, **args):

        # check the class of each
        assert all([args[i].__class__.__name__ == 'Vector' for i in args]), "Arguments must be Vector objects."

        # create the object instance
        super().__init__(**args)


    def __str__(self):
        return "\n".join([" ".join(["\n\nVector:\t ", i, "\n\n", self[i].__str__()]) for i in self])


    def __repr__(self):
        return self.__str__()


    def to_df(self, wide=False):
        """
        Return a dict of pandas DataFrames containing all the vectors data. The vectors are stored according to their
        type and each dataframe is generated in wide or long format according to the wide option.

        Input:
            wide: (bool)
                If False, each dataframe will be in "long format". The index and dimensions are stored as columns
                with names "Index_ZZZ" (for the index) and the "XXX.YYY_ZZZ" for the columns where "XXX" is the Vector,
                "YYY" indicates the dimension of the vector and "ZZZ" the unit of measurement.
                If True, the dataframe will be created in wide format. This means that each row will be one dimension
                of the vector. The first column will name the dimension, the second the unit of measurement and the
                others the values. These values will have column name in the form "KKK ZZZ" where "KKK" is one index
                value and "ZZZ" the index unit of measurement.

        Output:
            dc: (dict)
                a dict with dataframes.
        """

        # check the entered data
        assert wide or not wide, "'wide' must be a bool."

        # import the necessary packages
        from numpy import unique, append, arange
        from pandas import concat, DataFrame

        # get the indices pool for all types of data
        idxs = {}
        xunits = {}
        for i in self.keys():

            # generate a new key
            if not self[i].type in idxs.keys():
                idxs.update({self[i].type: self[i].index.to_numpy()})
                xunits.update({self[i].type: self[i].xunit})

            # update an old one
            else:
                idxs[self[i].type] = append(idxs[self[i].type], self[i].index.to_numpy())
                xunits[self[i].type] = append(xunits[self[i].type], self[i].xunit)

        # retain unique values
        idxs = {i: unique(idxs[i]) for i in idxs}
        xunits = {i: unique(xunits[i]) for i in xunits}

        # check units
        assert [len(xunits[i]) == 1 for i in xunits], "Multiple xunits have been found."

        # generate the dict containing all the types
        dfs = {i: DataFrame({'Index_' + xunits[i][0]: idxs[i]}, index=idxs[i]) for i in idxs}

        # decorate all the dataframes
        for i in self.keys():

            # get the new df
            new_df = self[i].to_df(False)

            # update the index
            new_df.index = new_df[new_df.columns[0]].values.flatten()

            # remove the first column
            new_df = new_df[new_df.columns[1:]]

            # update the columns
            new_df.columns = [((i + ".") if i != j.split("_")[0] else "") + j for j in new_df.columns]

            # add the new dataframe according to its type
            dfs[self[i].type] = concat((dfs[self[i].type], new_df), axis=1, sort=False)

        # manage the wide format
        if wide:
            for i in dfs:

                # transpose the dataframe
                dfs[i] = dfs[i][dfs[i].columns[1:]].T

                # update the columns
                dfs[i].columns = [str(j) + " " + xunits[i][0] for j in dfs[i].columns]

                # add the column with the remaining information
                dfs[i].insert(0, 'Unit', [j.split("_")[-1] for j in dfs[i].index.to_numpy()])
                dfs[i].insert(0, 'Dimension', [j.split(".")[-1].split("_")[0] for j in dfs[i].index.to_numpy()])
                dfs[i].insert(0, 'Vector', [j.split(".")[0].split("_")[0] for j in dfs[i].index.to_numpy()])

        # update the index
        for i in dfs: dfs[i].index = arange(dfs[i].shape[0])

        # return the dict
        return dfs


    def to_excel(self, file, wide=False):
        """
        Store the data of this object into an excel file. The vectors are stored according to their
        type and each dataframe is generated in wide or long format according to the wide option.

        Input:
            file: (str)
                the path where to store the data.

            wide: (bool)
                If False, each dataframe will be in "long format". The index and dimensions are stored as columns
                with names "Index_ZZZ" (for the index) and the "XXX.YYY_ZZZ" for the columns where "XXX" is the Vector,
                "YYY" indicates the dimension of the vector and "ZZZ" the unit of measurement.
                If True, the dataframe will be created in wide format. This means that each row will be one dimension
                of the vector. The first column will name the dimension, the second the unit of measurement and the
                others the values. These values will have column name in the form "KKK ZZZ" where "KKK" is one index
                value and "ZZZ" the index unit of measurement.
        """

        # import dependancies
        from pybiomech.utils import to_excel as xlsx

        # check the file
        assert file.__class__.__name__ == 'str', "'file' must be a str."
        assert wide or not wide, "'wide' must be a bool."

        # get the data
        dfs = self.to_df(wide)

        # store the data
        [xlsx(file, dfs[i], i) for i in dfs]


    def to_csv(self, path, sep=";", wide=False):
        """
        Store the data of this object into a folder containing one csv file collecting the vectors with the same type.
        The vectors are stored according to their type and each dataframe is generated in wide or long format according
        to the wide option.

        Input:
            path: (str)
                the path where to store the data.

            sep: (str)
                the separator to be used during the generation of the csv file(s).

            wide: (bool)
                If False, each dataframe will be in "long format". The index and dimensions are stored as columns
                with names "Index_ZZZ" (for the index) and the "XXX.YYY_ZZZ" for the columns where "XXX" is the Vector,
                "YYY" indicates the dimension of the vector and "ZZZ" the unit of measurement.
                If True, the dataframe will be created in wide format. This means that each row will be one dimension
                of the vector. The first column will name the dimension, the second the unit of measurement and the
                others the values. These values will have column name in the form "KKK ZZZ" where "KKK" is one index
                value and "ZZZ" the index unit of measurement.
        """

        # dependancies
        import os

        # check the file
        assert path.__class__.__name__ == 'str', "'file' must be a str."
        assert wide or not wide, "'wide' must be a bool."
        assert sep.__class__.__name__ == "str", "'str' must be a str object."

        # get the data
        dfs = self.to_df(wide)

        # ensure the path exist
        os.makedirs(path, exist_ok=True)

        # store the data
        [dfs[i].to_csv(path + "\\" + i + ".csv", sep=sep, index=False) for i in dfs]


    def from_excel(self, file, sheets=None, wide=False, types=None):
        """
        Add vectors to the current VectorDict from an excel file. The Excel file must have one sheet per type of
        vector. In addition, if the data are store in wide format:

        Otherwise, the data stored in long format require:
            - On each sheet, the index in the first column with header "Inder_ZZZ".
            - On each sheet, all the columns after the first must have header set as "XXX.YYY_ZZZ" where "XXX" is the
            name of the vector, "YYY" is the dimension and "ZZZ" the unit of measurement.

        Input:
            file: (str)
                a valid ".xlsx" or ".xls" file.
            sheets: (None or list or ndarray or str or int)
                - if "sheets" is a list or ndarray of "int",
                    they will be the number of the sheets being imported (0 will be the first). If "sheets" is a list
                    or ndarray of "str", they will be the name of the sheets to be imported.
                - if "sheets" is None,
                    the all sheets in the file are imported.
                - if "sheets" is an int,
                    only the i-th sheet will be imported (0 is the first sheet).
                - if "sheets" is a str,
                    it will be the name of the sheet to be imported.
            wide: (bool)
                If False, each sheet will be considered to have data stored in long format, otherwise the data will be
                uploaded considering each sheet according to the wide format.
            types: (list, None)
                a list of str files reflecting the type of data (i.e. the sheets names) that have to be imported. This
                parameter is useless in case wide is True. If None (default) all the sheets are tried to be imported.
        """

        # import the necessary packages
        from pybiomech.utils import from_excel, classcheck, extract
        from numpy import unique, arange

        # check the entered file
        classcheck(file, ["str"])
        assert file.split(".")[-1] in ['xlsx', 'xls'], "'file' must be an '.xlsx' or an '.xls' file."
        assert wide or not wide, "'wide' must be a bool."
        classcheck(types, ["list", "NoneType"])
        if types is not None: [classcheck(i, ["str"]) for i in types]

        # get the sheets as dict
        dfs = from_excel(file, sheets)

        # reframe each key
        if types is None: types = [i for i in dfs]
        for k in [i for i in dfs if i in types]:
            try:

                # import in wide format
                if wide:

                    # get columns with values
                    cols = [i for i in dfs[k].columns[3:]]

                    # get the index unit
                    xunit = unique([i.split(" ")[-1] for i in cols])

                    # check xunit
                    assert len(xunit) == 1, "Multiple xunits have been found."

                    # get the index
                    x = unique([float(i.split(" ")[0]) for i in cols])

                    # generate the vectors
                    for i in unique(dfs[k]['Vector'].values.flatten()):

                        # get the subset corresponding to the current vector
                        sub = extract(dfs[k], {'Vector': [i]})[0]

                        # get the dunit
                        dunit = unique(sub['Unit'].values.flatten())

                        # check xunit
                        assert len(dunit) == 1, "Multiple dunits have been found for vector " + i + "."

                        # store the vector
                        dt = {sub['Dimension'].values[j]: sub.iloc[j, cols].values.flatten()
                              for j in arange(sub.shape[0])}
                        self[i] = Vector(dt, x, str(xunit), str(dunit), str(k))

                # import data in long format
                else:

                    # get the index unit
                    xunit = dfs[k].columns[0].split("_")[-1]

                    # get the dunit
                    dunit = unique([i.split("_")[-1] for i in dfs[k].columns[1:]])
                    assert len(dunit) == 1, "multiple units dimensions have been found in " + k + "."
                    dunit = str(dunit[0])

                    # get the index
                    x = dfs[k][dfs[k].columns[0]].values.flatten()

                    # get the vectors
                    cols = dfs[k].columns[1:]
                    vctrs = [i.split(".")[0] if len(i.split(".")) > 1 else "_".join(i.split("_")[:-1]) for i in cols]
                    vctrs = unique(vctrs)
                    for v in vctrs:

                        # get the dimensions of each vector
                        vdata = {}
                        for j in cols:
                            if j[:len(v)] == v:
                                d = "_".join(j.split(".")[-1].split("_")[:-1]) if len(j.split(".")) > 1 else v
                                vdata[d] = dfs[k][((v + ".") if d != v else "") + d + "_" + dunit].values.flatten()

                        # store the Vector
                        self[v] = Vector(vdata, x, str(xunit), str(dunit), str(k))
            except Exception: pass


    def from_csv(self, file, sep=";"):
        """
        Add "Vector" objects to the current VectorDict Vectors collected from an ".csv" or ".txt" file.

        Input:
            file: (str)
                an existing ".txt" or ".csv" file.

            sep: (str)
                the separator used by the csv file.
        """

        # import the necessary packages
        from os.path import exists
        import numpy as np
        from pandas import read_csv

        # check the validity of the entered file
        assert exists(file), file + ' does not exist.'
        assert file.split(".")[-1] in np.array(['txt', 'csv']), file + ' must be a ".csv" or ".txt" file.'

        # read the file
        df = read_csv(file, sep=sep, index_col=False)
        cols = df.columns.to_numpy()

        # get the units
        dunit = np.unique([i.split("_")[-1] for i in cols[1:]])
        assert len(dunit) == 1, "multiple 'dunit' have been found."
        dunit = dunit[0]
        xunit = cols[0].split("_")[-1]
        typ = ".".join(file.split("\\")[-1].split(".")[:-1])

        # index
        idx = df[cols[0]].values.flatten().astype(float)

        # find vectors and its location
        vctrs = {}
        for i, v in enumerate(cols[1:]):
            seg = v.split(".")
            if len(seg) > 1:
                vec = ".".join(seg[:-1])
            else:
                seg = seg[0].split("_")
                vec = "_".join(seg[:-1]) if len(seg) > 2 else seg[0]
            try: vctrs[vec] = np.append(vctrs[vec], i + 1)
            except Exception: vctrs[vec] = np.array([i + 1])

        # extract each vector
        for vec in vctrs:

            # get the dimensions
            dims = {cols[i].split(".")[-1].split("_")[0]: df[cols[i]].values.flatten().astype(float)
                    for i in vctrs[vec]}

            # generate the new vector
            self[vec] = Vector(dims, idx, str(xunit), str(dunit), str(typ))


    def from_emt(self, file):
        """
        Add "Vector" objects to the current VectorDict Vectors collected from an ".emt" file.

        Input:
            file: (str)
                an existing ".emt" file.
        """

        # import the necessary packages
        from os.path import exists
        import numpy as np

        # check the validity of the entered file
        assert exists(file), file + ' does not exist.'
        assert file[-4:] == '.emt', file + ' must be an ".emt" file.'

        # read the file
        try:
            file = open(file)

            # get the lines of the file
            lines = [[j.strip() for j in i] for i in [i.split('\t') for i in file]]

        # something went wrong so close file
        except Exception: pass

        # close the file
        finally: file.close()

        # get the units
        dunit = lines[3][1]
        xunit = 's'

        # get the type
        type = lines[2][1]

        # get an array with all the variables
        V = np.array([i for i in lines[10] if i != ""]).flatten()

        # get the data names
        names = np.unique([i.split('.')[0] for i in V[2:] if len(i) > 0])

        # get the data values
        values = np.vstack([np.atleast_2d(np.array(i)[:-1]) for i in lines[11:-2]]).astype(float)

        # get the columns of interest
        cols = np.arange(np.argwhere(V == "Time").flatten()[0] + 1, len(V))

        # get the rows in the data to be extracted
        rows = np.argwhere(np.any(~np.isnan(values[:, cols]), 1)).flatten()
        rows = np.arange(np.min(rows), np.max(rows) + 1)

        # get time
        time = values[rows, 1].flatten()

        # generate a dataframe for each variable
        for v in names:

            # get the dimensions
            D = [i.split(".")[-1] for i in V if i.split(".")[0] == v]
            D = [""] if len(D) == 1 else D

            # get the data for each dimension
            K = {i if i != "" else v: values[rows, np.argwhere(V == v + (("." + i)
                                                                if i != "" else "")).flatten()].flatten() for i in D}

            # setup the output variable
            self[v] = Vector(K, time, xunit, dunit, type)
