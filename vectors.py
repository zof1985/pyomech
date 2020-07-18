


# DEPENDANCIES



import numpy as np
import pandas as pd
import warnings
import os
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.exceptions import ConvergenceWarning
from bokeh.plotting import *
from bokeh.layouts import *
from bokeh.models import *
from utils import *
from processing import *
warnings.filterwarnings("ignore", category=ConvergenceWarning)



class Vector(pd.DataFrame):
    """
    class representing an n-dimensional vector sampled over time.

    """   


    # class properties


    time_unit = ""
    dim_unit = ""
    type = ""
    _metadata = ["time_unit", "dim_unit", "type"]



    def cubic_spline_interpolation(self, x=None, n=None, plot=True):
        """
        return a vector after having cubic-spline interpolated its dimensions.

        Input:
            x:      (1D array)
                    the x coordinates to be used as new x-axis after the interpolation.

            n:      (int)
                    if x is None, n samples are used to generate the resulting interpolated vector.

            plot:   (bool)
                    if True, return a bokeh figure showing the outcomes of the interpolation.

        Output:
            V:      (Vector)
                    the interpolated vector

            p:      (bokeh.Figure optional)
                    the figure representing the outcomes of the interpolation.
        """

        # get x
        if x is None:
            x = np.linspace(self.index.to_numpy()[0], self.index.to_numpy()[-1], n)
        
        # get a copy of self
        V = self.loc[x]

        # apply the pyomech.processing.winter_residuals method to each dimension of the Vector
        plots = {}
        for i, v in enumerate(self.columns):
            if plot:
                V.loc[V.index, v], ps = cubic_spline_interpolation(y=self[v].values.flatten(),
                                                                      x_old=self.index_to_numpy(),
                                                                      x_new=x, plot=plot)
                plots[v] = ps

            else:
                V.loc[V.index, v] = cubic_spline_interpolation(y=self[v].values.flatten(),
                                                                  x_old=self.index.to_numpy(),
                                                                  x_new=x, plot=plot)

        # return the data
        if not plot:
            return V

        # generate the output figure
        figures = []
        for i, v in enumerate(plots):
            
            # get the figure corresponding to the current dimension
            f = plots[v].children[0]

            # adjust the axis labels
            f.xaxis.axis_label = self.time_unit
            f.yaxis.axis_label = self.dim_unit if i == 0 else ""

            # set the legend
            if i < len(plots) - 1:
                f.legend.visible = False
            else:
                f.legend.hide_policy = None
            
            # set the title
            f.title.text = v

            # add the columns to the figure output
            figures += [f]
            
        # return all
        return V, gridplot(figures, toolbar_location="right", merge_tools=True)



    def der1_winter(self):
        '''
        return the first derivative over time for the vector according to Winter 2009

        References:
            Winter DA.
                Biomechanics and Motor Control of Human Movement. Fourth Ed.
                Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
        '''
        x = np.atleast_2d(self.index.to_numpy()).T
        y = self.values
        d1 = (y[2:, :] - y[:-2, :]) / (x[2:, :] - x[:-2, :])
        dim_unit = self.dim_unit + ' * ' + self.time_unit + "^-1"
        return Vector(d1, index=x.flatten()[1:-1], time_unit=self.time_unit, dim_unit=dim_unit, type=self.type,
                      columns=self.columns)
    


    def der2_winter(self):
        '''
        return the second derivative over time for the vector according to Winter 2009

        References:
            Winter DA.
                Biomechanics and Motor Control of Human Movement. Fourth Ed.
                Hoboken, New Jersey: John Wiley & Sons Inc; 2009.
        '''
        d2 = self.loc[1:-1]
        t = np.vstack(np.atleast_2d([self.index.to_numpy() for i in self.columns])).T
        x = self.values
        dv = (x[2:] - x[1:-1]) / (t[2:] - t[1:-1]) - (x[1:-1] - x[:-2]) / (t[1:-1] - t[:-2])
        dt = (t[2:] - t[:-2]) * 0.5
        d2.loc[0:] = dv / dt
        d2.dim_unit = d2.dim_unit + ' * ' + d2.time_unit + "^-2"
        return d2



    def nanreplace(self, vectors={}, max_tr_data=1000, replacing_policy=None, plot=True, GridSearchKwargs={}, SVRKwargs={},
                   SVRMode=True):
        '''
        Use Support Vector Regression (SVR) to provide the coordinates of the missing samples in the current vector.

        Input:
            vectors: (VectorDict)
                list of reference vectors whoose coordinates can be used to train the SVR model.
            max_tr_data: (int)
                the maximum number of training data to be used. If the effective available number is lower than it,
                all the training data are used. Otherwise, the specified numebr is randomly sampled from the available
                pool.
            replacing_policy: (dict or None)
                How to replace the NaNs data. If a dict is provided, the NaNs contained in the dimensions of the vctor
                corresponding to any of the key contained in the provided dict will be replaced by the value
                associated to that key. That value can be a float or an int object. If None is provided or a key is
                not given for one or more dimensions of the vector, SVR is used to provide the NaNs replacement.
            GridSearchKwargs:
                parameters passed to the scikit-learn GridSearchCV class.
            SVRKwargs:
                other parameters passed to the SVR class.
            SVRMode:    (boolean)
                True means that SVR is used to estimate missing data.
                Otherwise missing data are filled via Cubic Spline interpolation

        Output:
            complete: (Vector)
                a new Vector equal to the current one but without missing data.
            replaced: (Vector)
                the Vector containing the missing data.
            fig: (optional)
                if plot is True, a bokeh plot is generated ready to be printed out.

        References:
            Smola A. J., Schölkopf B. (2004).
            A tutorial on support vector regression. Statistics and Computing, 14(3), 199–222.
        '''

        # get a copy of the current vector
        complete = self.copy()

        # get the missing values
        miss_idx = self.index[self.isna().any(1)].to_numpy()

        # replace missing data
        if len(miss_idx) > 0:

            # check the replacement policy
            if replacing_policy is None:
                replacing_policy = {i: None for i in self.columns}
            else:
                for i in self.columns:
                    if i not in np.array([j for j in replacing_policy]):
                        replacing_policy[i] = None

            # check which mode should be used
            if SVRMode:
                
                # SVR estimator options
                opt_SVRKwargs = {
                    "kernel": "rbf",
                    "gamma": "scale",
                    "tol": 1e-5,
                    "epsilon": 5e-4,  # i.e. 0.5 mm error.
                    "max_iter": 1e4
                    }
                opt_SVRKwargs.update(**SVRKwargs)

                # prepare the GridSearchCV options
                opt_GridSearchKwargs = {
                    "estimator": SVR(**opt_SVRKwargs),
                    "param_grid": {
                        "C": np.unique([i * (10 ** j) for i in np.arange(1, 11) for j in np.linspace(-10, -1, 10)])
                        },
                    "scoring": "neg_mean_absolute_error"
                    }
                opt_GridSearchKwargs.update(**GridSearchKwargs)

                # get the training dataset
                x = pd.DataFrame()
                for v in vectors:
                    v_pdf = vectors[v].loc[self.index]
                    v_pdf.columns = np.array(["_".join([v, i]) for i in vectors[v].columns])
                    x = pd.concat([x, v_pdf], axis=1, ignore_index=False)

                # exclude the features containing NaNs in the miss_idx data range
                valid_features = x.columns[~(x.loc[miss_idx].isna().any(0))]
                x = x[valid_features]

                # exclude the sets containing missing data from the training sets,
                # then get max_tr_data unique samples at random
                valid_sets = x.index[(~x.isna().any(1) & ~x.index.isin(miss_idx))].to_numpy()
                np.random.seed()
                unique_sets = valid_sets[np.unique(x.loc[valid_sets].values, axis=0, return_index=True)[1]]
                training_index = np.random.permutation(unique_sets)[:max_tr_data]
                training_set = x.loc[training_index]

                # grid searcher
                grid = GridSearchCV(**opt_GridSearchKwargs)

                # work on each vector dimensions separately
                for i, v in enumerate(self.columns):
                    if replacing_policy[v] is None:
                        
                        # get the best estimator
                        est = grid.fit(training_set.values, self.loc[training_index, v].values.flatten())

                        # predict the missing samples using the trained model
                        complete.loc[miss_idx, v] = est.best_estimator_.predict(x.loc[miss_idx].values)
                    
                    # replace missing data with the replacing policy value
                    else:
                        complete.loc[miss_idx, v] = replacing_policy[v]

            # Cubic Spline interpolation should be used
            else:

                # work on each vector dimensions separately
                for i, v in enumerate(self.columns):
                    if replacing_policy[v] is None:
                        
                        # get the list of complete datasets
                        x_old = self.index[~self.isna().any(1)].to_numpy()
                        x_new = complete.index.to_numpy()
                        y_old = complete.loc[x_old, v].values.flatten()
                        y_new = cubic_spline_interpolation(y=y_old, x_old=x_old, x_new=x_new)
                        complete.loc[miss_idx, v] = pd.DataFrame(y_new, index=x_new).loc[miss_idx].values.flatten()
                    
                    # replace missing data with the replacing policy value
                    else:
                        complete.loc[miss_idx, v] = replacing_policy[v]

        # get the replaced data
        replaced = complete.loc[miss_idx]
        

        # check if a plot has to be generated
        if not plot:
            return complete, replaced
        
        # generate one figure for each dimension
        plot_width = 300
        plots = []
        handles = {}
        for i, v in enumerate(self.columns):
                
            # generate the figure
            plots += [figure(width=plot_width, height=plot_width, title="Dimension: " + v)]
                
            # link the x_range
            if i > 0:
                plots[i].x_range = plots[0].x_range

            # edit the axes labels
            plots[i].xaxis.axis_label = self.time_unit
            plots[i].yaxis.axis_label = self.dim_unit if i == 0 else ""

            # plot the original data
            complete_idx = self.index[~self.isna().any(1)].to_numpy()
            original = self.loc[complete_idx]
            raw = plots[i].scatter(original.index.to_numpy(), original[v].values.flatten(), size=4, color="navy",
                                   alpha=0.5, marker="circle")
            try:
                handles['Raw'] = [raw]
            except Exception:
                handles['Raw'] += [raw]

            # plot the replaced data
            if len(miss_idx) > 0:
                rep = plots[i].scatter(replaced.index.to_numpy(), replaced[v].values.flatten(), size=4, color="red",
                                       marker="cross", alpha=0.5)
                try:
                    handles['Replaced'] = [rep]
                except Exception:
                    handles['Replaced'] += [rep]

            # edit the grids
            plots[i].xgrid.grid_line_alpha=0.3
            plots[i].ygrid.grid_line_alpha=0.3
            plots[i].xgrid.grid_line_dash=[5, 5]
            plots[i].ygrid.grid_line_dash=[5, 5]
            
        # add the legend to the rightmost plot
        if len(miss_idx) > 0:
            legend = Legend(items=[LegendItem(label=i, renderers=handles[i]) for i in handles], title="Legend")
            plots[-1].add_layout(legend)
            
        # put together all the plots
        p = gridplot(plots, ncols=len(self.columns), toolbar_location="right", merge_tools=True)

        # return all
        return complete, replaced, p



    def winter_cutoffs(self, f_num=100, f_max=None, segments=2, min_samples=2, plot=True, filt_fun=None,
                       filt_opt=None):
        """
        Calculate the optimal cutoff via a modified Winter (2009) approach.

        Input:
            f_num: (int)
                the number of frequencies to be tested within the (0, f_max) range to create the residuals curve of
                the Winter's residuals analysis approach.
            f_max: (int)
                the maximum filter frequency that is tested. If None, it is defined as the fs/4 of the Vector.
            segments: (int)
                the number of segments that can be used to fit the residuals curve in order to identify the best
                deflection point. It should be a value between 2 and 4.
            min_samples: (int)
                the minimum number of elements that have to be considered for each segment during the calculation of
                the best deflection point.
            plot: (bool)
                should a plot be returned?
            filt_fun: (function)
                the filter to be used for the analysis. If None, a Butterworth, low-pass, 4th order phase-corrected
                filter is used.
            filt_opt: (dict)
                the options for the filter.

        Output:
            cutoffs: (pandas.DataFrame)
                a pandas.DataFrame with one value per column defining the optimal cutoff frequency.
            SSEs: (pandas.DataFrame)
                a pandas.DataFrame with the selected frequencies as index and the Sum of Squared Residuals as columns.
            fig: (bokeh.figure)
                a figure is provided if plot is True
        """
                    
        # initialize the output values
        SSEs = pd.DataFrame()
        cutoffs = pd.DataFrame()
        plots = {}

        # apply the pyomech.processing.winter_residuals method to each dimension of the Vector
        for i, v in enumerate(self.columns):
            if plot:
                cut, sse, plt = winter_residuals(x=self[v].values.flatten(),
                                                    fs=self.sampling_frequency,
                                                    f_num=f_num,
                                                    f_max=f_max,
                                                    segments=segments,
                                                    min_samples=min_samples,
                                                    filt_fun=filt_fun,
                                                    filt_opt=filt_opt,
                                                    plot=plot)
                plots[v] = plt

            else:
                cut, sse = winter_residuals(x=self[v].values.flatten(),
                                               fs=self.sampling_frequency,
                                               f_num=f_num,
                                               f_max=f_max,
                                               segments=segments,
                                               min_samples=min_samples,
                                               filt_fun=filt_fun,
                                               filt_opt=filt_opt,
                                               plot=plot)

            # merge the output
            cutoffs[v] = pd.Series(cut)
            SSEs = pd.concat([SSEs, sse], axis=1, ignore_index=False)

        # return the data
        if not plot:
            return cutoffs, SSEs

        # generate the output figure
        for i, v in enumerate(plots):
                
            # remove unnecessary axis labels
            if i > 0:
                plots[v].yaxis.axis_label = ""
            
            else:

                # adjust the y axis unit
                plots[v].yaxis.axis_label = plots[v].yaxis.axis_label + " (" + self.dim_unit + ")"
            
            # remove unnecessary legends
            if i < len(plots) - 1:
                plots[v].legend.visible = False

            # set the title
            plots[v].title.text = "Dimension: " + v
            
        # put together all the plots
        p = gridplot([plots[v] for v in plots], ncols=len(self.columns), toolbar_location="right", merge_tools=True)

        # return all
        return cutoffs, SSEs, p



    def butt_filt(self, cutoffs, order=4, type='lowpass', phase_corrected=True, plot=True):
        """
        a convenient wrapper of the pyomech.processing.butt_filt method which applies to all the dimensions of the
        vector.

        Input:
            cutoffs:            (float or dict)
                                the cut-off for the filter. If a dict is provided, the keys corresponding to the vector
                                dimensions must contain the required filter cutoff.

            order:              (int)
                                the order of the filter.
            
            type:               (str)
                                the type of the filter.

            phase_corrected:    (bool)
                                should the filter be applied a second time with the signal in reverse order?

            plot:               (bool)
                                if True a bokeh figure is returned in addition to the filtered data vector.
          
        Output:
            vf:                 (Vector)
                                the Vector containing the filtered data.

            fig:                (bokeh.figure, optional)
                                a bokeh figure showing the effect of the filter.
        """
                    
        # initialize the output values
        vf = self.copy()
        plots = {}

        # handle the cutoffs
        if cutoffs.__class__.__name__[:5] == "float" or cutoffs.__class__.__name__[:3] == "int":
            cutoffs = {i: cutoffs for i in self.columns}

        # apply the pyomech.processing.winter_residuals method to each dimension of the Vector
        for i, v in enumerate(self.columns):
            if plot:
                yf, plt = butt_filt(self[v].values.flatten(), cutoffs[v], self.sampling_frequency, order, type,
                                       phase_corrected, plot)
                plots[v] = plt

            else:
                yf = butt_filt(self[v].values.flatten(), cutoffs[v], self.sampling_frequency, order, type,
                                  phase_corrected, plot)

            # merge the output
            vf.loc[vf.index, v] = yf

        # return the data
        if not plot:
            return vf

        # generate the output figure
        figures = []
        for i, v in enumerate(plots):
            
            # each plot is a column gridplot of 2 figures that have to be managed separately
            col_fig = []
            for j in np.arange(len(plots[v].children[0].children)):
                
                # remove unnecessary axis labels
                if i > 0:
                    plots[v].children[0].children[j][0].yaxis.axis_label = ""
                
                # adjust the label
                else:
                    if j == 0:
                        txt = "Signal amplitude (" + self.dim_unit + ")"
                        plots[v].children[0].children[j][0].yaxis.axis_label = txt

                # adjust the time-domain data X-coordinates
                if j == 0:
                    for k in np.arange(len(plots[v].children[0].children[j][0].renderers)):
                        val = plots[v].children[0].children[j][0].renderers[k].data_source.data['x']
                        val = val / self.sampling_frequency
                        plots[v].children[0].children[j][0].renderers[k].data_source.data['x'] = val
                        plots[v].children[0].children[j][0].xaxis.axis_label = "Time (" + self.time_unit + ")"
            
                # remove unnecessary legends
                if i < len(plots) - 1:
                    plots[v].children[0].children[j][0].legend.visible = False

                # set the title
                if j == 0:
                    plots[v].children[0].children[j][0].title.text = "Dimension: " + v + " (Time domain)"
                else:
                    plots[v].children[0].children[j][0].title.text = "Dimension: " + v + " (Frequency domain)"

                # add the figure to the column
                col_fig += [plots[v].children[0].children[j][0]]

            # add the columns to the figure output
            figures += [col_fig]
            
        # transpose the figures matrix
        figures = list(map(list, zip(*figures)))
        p = gridplot(figures, toolbar_location="right", merge_tools=True)

        # return all
        return vf, p
    


    @staticmethod
    def angle(A, B, C):
        """
        return the angle ABC using the Cosine theorem.

        Input:
            A:  (Vector)
                The coordinates of one point.

            B:  (Vector)
                The coordinates of the point over which the angle has to be calculated.
            
            C:  (Vector)
                The coordinates of the third point.
        
        Output:
            K:  (Vector)
                A 1D vector containing the result of:
                                     2          2            2
                           /  (A - B)  +  (C - B)  -  (A - C)  \
                    arcos | ----------------------------------- |
                           \      2 * (A - B) * (C - B)        /
        """

        # ensure all entered parameters are vectors
        assert match(A, B, C), "'A', 'B' and 'C' must be Vectors with the same index and columns."

        # get a, b and c
        a = (A - B).module
        b = (C - B).module
        c = (A - C).module

        # return the angle
        return ((a ** 2 + b ** 2 - c ** 2) / (2 * a * b).values).apply(np.arccos, 0)



    @staticmethod
    def intercept(A, B, C):
        """
        return the Vector X such as the segment CX will have minimum distance to the segment AB. 
        
        Input:
            A, B:   (Vector)
                    coordinates of the vectors defining the segment from which minimum distance
                    has to be calculated. 
            
            C:      (Vector)
                    the coordinates of the point from which minimum distance to the segment AB
                    has to be calculated.
        
        Output:
            X:      (Vector)
                    the coordinates of the vector along the segment AB minimizing the CX length.
        """

        # ensure all entered parameters are vectors
        assert match(A, B, C), "'A', 'B' and 'C' must be Vectors with the same index and columns."

        # get the multiplier to be used for calculating the distance.
        t = (A - C).dot(B - A) / ((B - A).module.values ** 2)

        # get X
        return A + (B - A) * t


    
    @property
    def module(self):
        """
        Get the module of the vector.
        """
        return Vector((self ** 2).sum(1).values.flatten() ** 0.5,
                      index=self.index,
                      columns=["|" + " + ".join(self.columns) + "|"])
     


    @property
    def sampling_frequency(self):
        return 1. / np.mean(np.diff(self.index.to_numpy()))

    

    @staticmethod
    def match(*args, **kwargs):
        """
        check if the entered objects are instance of Vector or pandas.DataFrame.
        If more than one parameter is provided, check also that all the entered objects have the 
        same columns and indices.

        Output:
            C:  (bool)
                True if B is a Vector or a pandas.DataFrame with the same columns and index of self.
                False, otherwise.
        """

        # get the elements entered
        objs = [i for i in args] + [kwargs[i] for i in kwargs]

        # check if all elements are instance of Vector or DataFrame
        for obj in objs:
            if not isinstance(obj, (Vector, pd.DataFrame)):
                return False
        
        # check the columns and index of all objs
        IX = objs[0].index.to_numpy()
        CL = objs[0].columns.to_numpy()
        SH = objs[0].shape
        for obj in objs:
            OI = obj.index.to_numpy()
            OC = obj.columns.to_numpy()
            col_check = np.all([i in OC for i in CL])
            idx_check = np.all([i in OI for i in IX])
            shp_check = np.all([i == j for i, j in zip(obj.shape, SH)])
            if not np.all([col_check, idx_check, shp_check]):
                return False
        return True



    def dot(self, B):
        """
        return a vector being the dot product between self and B.
        
        Input:
            B:  (Vector, numpy.ndarray, pandas.DataFrame)
                the object to be multiplied with.
        
        Output:
            D:  (Vector)
                the dot product of self with B.
        """

        # handle the case B is a pandas.DataFrame or a Vector with the same dimensions of self
        if match(self, B):
            new_column = " + ".join(self.columns.to_numpy())
            values = [i.dot(j.T) for i, j in zip(self.values, B.values)]
            return Vector(values, index=self.index, columns=[new_column]).__finalize__(self)
        
        # use standard pandas dot
        return super(Vector, self).dot(B).__finalize__(self)



    @staticmethod
    def read_csv(*args, **kwargs):
        """
        return the Vector from a "csv". The file is formatted having a column named "Index_ZZZ" and the others as
        "XXX|YYY_ZZZ" where:
            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit

        Outpu:
            v:  (Vector)
                the imported vector.
        """
        return Vector.from_csv(*args, **kwargs)



    @staticmethod
    def from_df(df):
        """
        return the Vector from a pandas DataFrame. The df is formatted having a column named "Index_ZZZ" and the
        others as "XXX|YYY_ZZZ" where:
            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit

        Output:
            v:  (Vector)
                the imported vector.
        """

        # get the index 
        idx_col = [i for i in df.columns if "_".join(i.split("_")[:-1])][0]
        idx_val = df[idx_col].values.flatten()

        # get the time_unit
        time_unit = idx_col.split("_")[-1]

        # remove the index column
        df = df[[i for i in df.columns if i != idx_col]]

        # get the vector type
        typ = np.unique(["|".join(i.split("|")[:-1]) for i in df.columns])
        txt = "No vector type has been found" if len(typ) == 0 else str(len(typ)) + " vector types have been found."
        assert len(typ) == 1, txt
        typ = typ[0]

        # get the dim_unit
        uni = np.unique([i.split("_")[-1] for i in df.columns])
        txt = "No unit type has been found" if len(uni) == 0 else str(len(uni)) + " dimension units have been found."
        assert len(uni) == 1, txt
        uni = uni[0]

        # update the columns
        df.columns = ["_".join(i.split("|")[-1].split("_")[:-1]) for i in df.columns]

        # get the vector
        return Vector(df.to_dict("list"), index=idx_val, time_unit=time_unit, dim_unit=uni, type=typ)



    @staticmethod
    def from_csv(*args, **kwargs):
        """
        return the Vector from a "csv". The file is formatted having a column named "Index_ZZZ" and the others as
        "XXX|YYY_ZZZ" where:
            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit

        Output:
            v:  (Vector)
                the imported vector.
        """
        return Vector.from_df(pd.read_csv(*args, **kwargs))



    @staticmethod
    def from_excel(file, sheet, *args, **kwargs):
        """
        return the Vector from an excel file. The file is formatted having a column named "Index_ZZZ" and the others as
        "XXX|YYY_ZZZ" where:
            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit

        Input:
            file:   (str)
                    the path to the file
            
            sheet:  (str)
                    the sheet to be imported

            args, kwargs: additional parameters passed to pandas.read_excel

        Output:
            v:  (Vector)
                the imported vector.
        """
        return Vector.from_df(from_excel(file, sheet, *args, **kwargs)[sheet])



    def to_df(self):
        """
        Store the Vector into a "pandas DataFrame" formatted having a column named "Index_ZZZ" and the others as
        "XXX|YYY_ZZZ" where:
            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit
        """
        
        # create the Vector df
        v_df = pd.DataFrame(self.values, columns=[self.type + "|" + i + "_" + self.dim_unit for i in self.columns])
        
        # add the index column
        v_df.insert(0, 'Index_' + self.time_unit, self.index.to_numpy())

        # return the df
        return v_df



    def to_csv(self, file, **kwargs):
        """
        Store the Vector into a "csv". The file is formatted having a column named "Index_ZZZ" and the others as
        "XXX|YYY_ZZZ" where:
            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit

        Input:
            file: (str)
                the file path.
        """
        
        # ensure the file can be stored
        os.makedirs(lvlup(file), exist_ok=True)

        # store the output data
        try:
            kwargs.pop('index')
        except Exception:
            pass
        try:
            kwargs.pop('path')
        except Exception:
            pass
        self.to_df().to_csv(file, index=False, **kwargs)



    def to_excel(self, file, sheet="Sheet1", new_file=False):
        """
        Store the Vector into an excel file sheet. The file is formatted having a column named "Index_ZZZ" and the
        others as "XXX|YYY_ZZZ" where:
            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit

        Input:
            file:       (str)
                        the file path.

            sheet:      (str or None)
                        the sheet name.

            new_file:   (bool)
                        should a new file be created rather than adding the current vector to an existing one?
        """
        
        to_excel(file, self.to_df(), sheet, new_file)



    # SUBCLASSED METHODS


    
    def __init__(self, *args, **kwargs):
        
        # remove special class objects
        props = {}
        for prop in self._metadata:
            try:
                props[prop] = kwargs.pop(prop)
            except Exception:
                pass
        
        # handle Series props
        ser_props = {}
        for prop in ["name", "fastpath"]:
            try:
                ser_props[prop] = kwargs.pop(prop)
            except Exception:
                pass

        # generate the pandas object
        if len(ser_props) > 0:
            super(Vector, self).__init__(pd.Series(*args, **ser_props))
        else:
            super(Vector, self).__init__(*args, **kwargs)
        
        # add the extra features
        for prop in props:
            setattr(self, prop, props[prop])



    def __finalize__(self, other, method=None, **kwargs):
        """propagate metadata from other to self """

        # merge operation: using metadata of the left object
        if method == "merge":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, getattr(self, name)))

        # concat operation: using metadata of the first object
        elif method == "concat":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, getattr(self, name)))

        # any other condition
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, getattr(self, name)))
        return self



    @property
    def _constructor(self):
        return Vector



    @property
    def _constructor_sliced(self):
        return Vector



    @property
    def _constructor_expanddim(self):
        return Vector



    def __str__(self):
        out = pd.DataFrame(self)
        return out.__str__() + "\n".join(["\nAttributes:", "\ttype:\t\t" + self.type,
                                          "\ttime_unit:\t" + self.time_unit,
                                          "\tdim_unit:\t" + self.dim_unit])



    def __repr__(self):
        return self.__str__()



    # MATH AND LOGICAL OPERATORS
    # basically each operator is forced to cast the __finalize__ method to retain the _metadata properties



    def __add__(self, *args, **kwargs):
        return super(Vector, self).__add__(*args, **kwargs).__finalize__(self)



    def __sub__(self, *args, **kwargs):
        return super(Vector, self).__sub__(*args, **kwargs).__finalize__(self)



    def __mul__(self, *args, **kwargs):
        return super(Vector, self).__mul__(*args, **kwargs).__finalize__(self)



    def __floordiv__(self, *args, **kwargs):
        return super(Vector, self).__floordiv__(*args, **kwargs).__finalize__(self)



    def __truediv__(self, *args, **kwargs):
        return super(Vector, self).__truediv__(*args, **kwargs).__finalize__(self)



    def __mod__(self, *args, **kwargs):
        return super(Vector, self).__mod__(*args, **kwargs).__finalize__(self)



    def __pow__(self, *args, **kwargs):
        return super(Vector, self).__pow__(*args, **kwargs).__finalize__(self)



    def __and__(self, *args, **kwargs):
        return super(Vector, self).__and__(*args, **kwargs).__finalize__(self)



    def __xor__(self, *args, **kwargs):
        return super(Vector, self).__xor__(*args, **kwargs).__finalize__(self)



    def __or__(self, *args, **kwargs):
        return super(Vector, self).__or__(*args, **kwargs).__finalize__(self)



    def __iadd__(self, *args, **kwargs):
        return super(Vector, self).__iadd__(*args, **kwargs).__finalize__(self)



    def __isub__(self, *args, **kwargs):
        return super(Vector, self).__isub__(*args, **kwargs).__finalize__(self)



    def __imul__(self, *args, **kwargs):
        return super(Vector, self).__imul__(*args, **kwargs).__finalize__(self)



    def __idiv__(self, *args, **kwargs):
        return super(Vector, self).__idiv__(*args, **kwargs).__finalize__(self)



    def __ifloordiv__(self, *args, **kwargs):
        return super(Vector, self).__ifloordiv__(*args, **kwargs).__finalize__(self)



    def __imod__(self, *args, **kwargs):
        return super(Vector, self).__imod__(*args, **kwargs).__finalize__(self)



    def __ilshift__(self, *args, **kwargs):
        return super(Vector, self).__ilshift__(*args, **kwargs).__finalize__(self)



    def __irshift__(self, *args, **kwargs):
        return super(Vector, self).__irshift__(*args, **kwargs).__finalize__(self)



    def __iand__(self, *args, **kwargs):
        return super(Vector, self).__iand__(*args, **kwargs).__finalize__(self)



    def __ixor__(self, *args, **kwargs):
        return super(Vector, self).__ixor__(*args, **kwargs).__finalize__(self)



    def __ior__(self, *args, **kwargs):
        return super(Vector, self).__ior__(*args, **kwargs).__finalize__(self)



    def __neg__(self, *args, **kwargs):
        return super(Vector, self).__neg__(*args, **kwargs).__finalize__(self)



    def __pos__(self, *args, **kwargs):
        return super(Vector, self).__pos__(*args, **kwargs).__finalize__(self)



    def __abs__(self, *args, **kwargs):
        return super(Vector, self).__abs__(*args, **kwargs).__finalize__(self)



    def __invert__(self, *args, **kwargs):
        return super(Vector, self).__invert__(*args, **kwargs).__finalize__(self)



    def __complex__(self, *args, **kwargs):
        return super(Vector, self).__complex__(*args, **kwargs).__finalize__(self)



    def __int__(self, *args, **kwargs):
        return super(Vector, self).__int__(*args, **kwargs).__finalize__(self)



    def __long__(self, *args, **kwargs):
        return super(Vector, self).__long__(*args, **kwargs).__finalize__(self)



    def __float__(self, *args, **kwargs):
        return super(Vector, self).__float__(*args, **kwargs).__finalize__(self)



    def __oct__(self, *args, **kwargs):
        return super(Vector, self).__oct__(*args, **kwargs).__finalize__(self)



    def __hex__(self, *args, **kwargs):
        return super(Vector, self).__hex__(*args, **kwargs).__finalize__(self)



    def __lt__(self, *args, **kwargs):
        return super(Vector, self).__lt__(*args, **kwargs).__finalize__(self)



    def __le__(self, *args, **kwargs):
        return super(Vector, self).__le__(*args, **kwargs).__finalize__(self)



    def __eq__(self, *args, **kwargs):
        return super(Vector, self).__eq__(*args, **kwargs).__finalize__(self)



    def __ne__(self, *args, **kwargs):
        return super(Vector, self).__ne__(*args, **kwargs).__finalize__(self)



    def __ge__(self, *args, **kwargs):
        return super(Vector, self).__ge__(*args, **kwargs).__finalize__(self)



    def __gt__(self, *args, **kwargs):
        return super(Vector, self).__gt__(*args, **kwargs).__finalize__(self)



class VectorDict(dict):
    """
    Create a dict of "Vector" object(s). It is a simple wrapper of the "dict" class object with additional methods.

    Input:
        args: (objects)
            objects of class vectors.
    """



    def to_csv(self, path, **kwargs):
        """
        store pandas.DataFrames containing the vectors formatted as: "XXX|YYY_ZZZ".
                
        where:
            'XXX' is the type of the vector
            'YYY' is the dimension of the vector
            'ZZZ'  if the dim_unit.
        
        In addition, the first column will be the index of the vectors.
        """

        # remove the filename from kwargs
        try:
            kwargs['path'] = None
        except Exception:
            pass

        # store all Vectors
        for v in self.keys():
            self[v].to_csv(os.path.sep.join([path, v + ".csv"]), **kwargs)
    


    def to_excel(self, path, new_file=False):
        """
        store an excel file containing the vectors formatted as: "XXX|YYY_ZZZ".
                
        where:
            'XXX' is the type of the vector
            'YYY' is the dimension of the vector
            'ZZZ'  if the dim_unit.
        
        In addition, the first column will be the index of the vectors.
        """
           
        # check if a new file must be created
        if new_file:
            os.remove(path)

        # store all Vectors
        [self[v].to_excel(path, v) for v in self]



    @staticmethod
    def from_csv(path, **kwargs):
        """
        Create a "VectorDict" object from a "csv" or "txt" file.

        Input:
            path: (str)
                an existing ".csv" or "txt" file or a folder containing csv files. The files must contain 1 column
                named "Index_ZZZ" and the others as "WWW:XXX|YYY_ZZZ" where:
                    'WWW' is the type of the vector
                    'XXX' is the name of the vector
                    'YYY' is the dimension of the vector
                    'ZZZ'  if the dim_unit.
        """

        # control the kwargs
        try:
            kwargs['index_col'] = False
        except Exception:
            pass

        # get the output dict
        vd = VectorDict()

        # check if the path is a file or a folder and populate the VectorDict accordingly
        if os.path.isfile(path):
            vd[".".join(path.split(os.path.sep)[-1].split(".")[:-1])] = Vector.from_csv(path, **kwargs)
        else:
            for i in get_files(path, ".csv", False):
                vd[".".join(i.split(os.path.sep)[-1].split(".")[:-1])] = Vector.from_csv(i, **kwargs)  

        # return the dict
        return vd



    @staticmethod
    def from_excel(path, sheets=None, exclude_errors=True):
        """
        Create a "VectorDict" object from an excel file.

        Input:
            path:           (str)
                            an existing excel file. The sheets must contain 1 column named "Index_ZZZ" and the
                            others as "WWW:XXX|YYY_ZZZ" where:
                                'WWW' is the type of the vector
                                'XXX' is the name of the vector
                                'YYY' is the dimension of the vector
                                'ZZZ'  if the dim_unit.

            sheets:         (str, list or None)
                            the sheets to be imported. In None, all sheets are imported.

            exclude_errors: (bool)
                            If a sheet generates an error during the import would you like to skip it and import the
                            others?
        
        Output:
            a new VectorDict with the imported vectors.
        """

        vd = VectorDict()

        # get the sheets
        dfs = from_excel(path, sheets)

        # import the sheets
        for i in dfs:
            if exclude_errors:
                try:
                    vd[i] = Vector.from_df(dfs[i])
                except Exception:
                    pass
            else:
                vd[i] = Vector.from_df(dfs[i])

        # return the dict
        return vd


    
    @staticmethod
    def from_emt(file):
        """
        Create a "VectorDict" object from a ".emt" file.

        Input:
            file: (str)
                an existing ".emt" file.
        """

        # check the validity of the entered file
        assert os.path.exists(file), file + ' does not exist.'
        assert file[-4:] == '.emt', file + ' must be an ".emt" file.'

        # read the file
        try:
            file = open(file)

            # get the lines of the file
            lines = [[j.strip() for j in i] for i in [i.split('\t') for i in file]]

        # something went wrong so close file
        except Exception:
            pass

        # close the file
        finally:
            file.close()

        # get the output VectorDict
        vd = VectorDict()

        # get the units
        dim_unit = lines[3][1]
        time_unit = 's'

        # get the type
        type = lines[2][1]

        # get an array with all the variables
        V = np.array([i for i in lines[10] if i != ""]).flatten()

        # get the data names
        names = np.unique([i.split('.')[0] for i in V[2:] if len(i) > 0])

        # get the data values (now should work)
        values = np.vstack([np.atleast_2d(i[:len(V)]) for i in lines[11:-2]]).astype(float)

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
            K = {i if i != "" else v: values[rows,
                                             np.argwhere(V == v + (("." + i)
                                                         if i != "" else "")).flatten()].flatten() for i in D}

            # setup the output variable
            vd[v] = Vector(K, index=time, time_unit=time_unit, dim_unit=dim_unit, type=type)
        
        # return vd
        return vd


   
    # SUBCLASSED METHODS
    


    def __init__(self, *args, **kwargs):
        super(VectorDict, self).__init__(*args, **kwargs)
        self.__finalize__


        
    # this method is casted at the end of each manipulating action in order to check that only Vector objects
    # are stored into the dict
    @property
    def __finalize__(self):
        for i in self.keys():
            assert self[i].__class__.__name__ == "Vector", i + " is not a Vector object"
        return self



    def __str__(self):
        return "\n".join([" ".join(["\n\nVector:\t ", i, "\n\n", self[i].__str__()]) for i in self.keys()])



    def __repr__(self):
        return self.__str__()



    def __setitem__(self, *args, **kwargs):
        super(VectorDict, self).__setitem__(*args, **kwargs)
        self.__finalize__



    def __setattr__(self, *args, **kwargs):
        super(VectorDict, self).__setattr__(*args, **kwargs)
        self.__finalize__