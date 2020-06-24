


########    IMPORTS    ########



import numpy as np
from .utils import classcheck, lvlup, from_excel
import pandas as pd



########    METHODS    ########



def geometric_mean(X, axis=0):
    """
    return the geometric mean along the defined axis of the dataset X.
    
    Input:
        X:      (ndarray)
                a numpy ndarray.

        axis:   (int)
                the axis along which the geometric mean must be calculated.

    Output:
        G:      (ndarray)
                the geometric mean calculated along the provided axis.
    """

    # get the number of arrays
    N = X.shape[axis]

    # return the geometric mean
    return np.prod(X, axis) ** (1 / N)



########    CLASSES    ########




class StateSpace(pd.DataFrame):
    """
    class representing the state of a system sampled over time.

    """   



    # SPECIFIC METHODS AND PROPERTIES



    @property
    def permutation_entropy(self):
        """
        """



    # CLASS PROPERTIES


    index_unit = ""
    space_units = []
    _metadata = ["time_unit", "space_units"]

  

    # IMPORT-EXPORT METHODS



    @staticmethod
    def from_1D(y, x=None, delays=True, from_index_unit="", space_unit=""):
        """
        Create a StateSpace with the specified number of dimensions from a 1D array (e.g. a time-series) using the
        method of delays.

        Input:
            y:          (1D array)
                        the signal to be used for the StateSpace reconstruction.

            x:          (1D array or None)
                        if provided, the array of indices for each sample in y.

            delays:     (list, 1D array or None)
                        a list of integers defining the delay to be applied for each embedding dimension.
                        If not provided, the Nichkawde 2013 approach is used to optimally reconstruct the StateSpace
                        from data.

            index_unit: (str)
                        The label defining the unit of measurement of the data provided by the x array.

            space_unit: (str)
                        The label defining the unit of measurement of the data provided by the y array.

        Output:
            S:          (StateSpace)
                        The StateSpace object corresponding to the provided data.
           
        References:
            Nichkawde C. (2013) Optimal state-space reconstruction using derivatives on projected manifold.
                PHYSICAL REVIEW E 87, 022905.
        """

        


    

    @staticmethod
    def read_csv(*args, **kwargs):
        """
        return the StateSpace from a "csv". The file is formatted having a column named "Index_ZZZ" and the others as
        "YYY_ZZZ" where:
            'YYY' the dimension of the Space
            'ZZZ' the corresponding space_unit

        Output:
            v:  (StateSpace)
                the imported StateSpace object.
        """
        return StateSpace.from_csv(*args, **kwargs)



    @staticmethod
    def from_df(df):
        """
        return the StateSpace from a "csv". The file is formatted having a column named "Index_ZZZ" and the others as
        "YYY_ZZZ" where:
            'YYY' the dimension of the Space
            'ZZZ' the corresponding space_unit

        Output:
            v:  (StateSpace)
                the imported StateSpace object.
        """

        # get the index 
        idx_col = [i for i in df.columns if "_".join(i.split("_")[:-1])][0]
        idx_val = df[idx_col].values.flatten()

        # get the index_unit
        index_unit = idx_col.split("_")[-1]

        # remove the index column
        df = df[[i for i in df.columns if i != idx_col]]

        # get the space_units
        uni = [i.split("_")[-1] for i in df.columns]
        
        # update the columns
        df.columns = [i[:(len(i) - len(v) - 1)] for i, v in zip(df.columns, uni)]

        # get the StateSpace
        return StateSpace(df.to_dict("list"), index=idx_val, index_unit=index_unit, space_units=uni)



    @staticmethod
    def from_csv(*args, **kwargs):
        """
        return the StateSpace from a "csv". The file is formatted having a column named "Index_ZZZ" and the others as
        "YYY_ZZZ" where:
            'YYY' the dimension of the Space
            'ZZZ' the corresponding space_unit

        Output:
            v:  (StateSpace)
                the imported StateSpace object.
        """
        return StateSpace.from_df(pd.read_csv(*args, **kwargs))



    @staticmethod
    def from_excel(file, sheet):
        """
        return the StateSpace from an excel file. The file is formatted having a column named "Index_ZZZ" and the
        others as "YYY_ZZZ" where:
            'YYY' the dimension of the StateSpace
            'ZZZ' the corresponding space_unit

        Input:
            file:   (str)
                    the path to the file
            
            sheet:  (str)
                    the sheet to be imported

            args, kwargs: additional parameters passed to pandas.read_excel

        Output:
            v:  (StateSpace)
                the imported StateSpace object.
        """
        return StateSpace.from_df(from_excel(file, sheet, *args, **kwargs)[sheet])



    def to_df(self):
        """
        Store the StateSpace into a "pandas DataFrame" formatted having a column named "Index_ZZZ" and the others as
        "YYY_ZZZ" where:
            'YYY' the dimension of the StateSpace object
            'ZZZ' the corresponding space unit
        """
        
        # create the Vector df
        v_df = pd.DataFrame(self.values, columns=[i + "_" + v for i, v in zip(self.columns, self.space_units)])
        
        # add the index column
        v_df.insert(0, 'Index_' + self.index_unit, self.index.to_numpy())

        # return the df
        return v_df



    def to_csv(self, file, **kwargs):
        """
        Store the StateSpace into a "csv" file such as the resulting table will have a column named "Index_ZZZ" and
        the others as "YYY_ZZZ" where:
            'YYY' the dimension of the StateSpace object
            'ZZZ' the corresponding space unit
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
        Store the StateSpace into an excel file such as the resulting table will have a column named "Index_ZZZ" and
        the others as "YYY_ZZZ" where:
            'YYY' the dimension of the StateSpace object
            'ZZZ' the corresponding space unit

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
            super(StateSpace, self).__init__(pd.Series(*args, **ser_props))
        else:
            super(StateSpace, self).__init__(*args, **kwargs)
        
        # add the extra features
        for name in props:

            # ensure index_unit is a str object
            if name == "index_unit":
                value = str(np.array([props[name]]).flatten()[0])

            # ensure spase_units has one value per dimension of self
            elif name == "space_units":
                value = np.array([props[name]]).flatten()
                if len(value) != self.shape[1]:
                    txt = "space_units must have len " + str(self.shape[1]) + " but len " + len(value) + " was found."
                    raise AttributeError(txt)
            
            # no other properties should be passed
            else:
                raise AttributeError(name + " is not a valid property for a StateSpace class object.")

            # add the property
            setattr(self, prop, props[prop])

        # if no unit properties have been passed. Initialize them as empty
        if "time_unit" not in [i for i in props]:
            self.time_unit = ""
        if "space_units" not in [i for i in  props]:
            self.space_units = ["" for i in np.arange(self.shape[1])]



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
        return StateSpace



    @property
    def _constructor_sliced(self):
        return StateSpace



    @property
    def _constructor_expanddim(self):
        return StateSpace


    
    def __str__(self):
        out = pd.DataFrame(self)

        # get the formatted index
        try:
            ix = np.array([" ".join(["{:0.3f}".format(i), self.index_unit]) for i in out.index.to_numpy()])
        except Exception:
            ix = np.array([" ".join([str(i), self.index_unit]) for i in out.index.to_numpy()])
        out.index = ix

        # get the formatted columns
        out.columns = [" ".join([out.columns[i], self.space_units[i]]) for i in np.arange(out.shape[1])]

        # print out
        return out.__str__()



    def __repr__(self):
        return self.__str__()
    

    
    # MATH AND LOGICAL OPERATORS
    # basically each operator is forced to cast the __finalize__ method to retain the _metadata properties



    def __add__(self, *args, **kwargs):
        return super(StateSpace, self).__add__(*args, **kwargs).__finalize__(self)



    def __sub__(self, *args, **kwargs):
        return super(StateSpace, self).__sub__(*args, **kwargs).__finalize__(self)



    def __mul__(self, *args, **kwargs):
        return super(StateSpace, self).__mul__(*args, **kwargs).__finalize__(self)



    def __floordiv__(self, *args, **kwargs):
        return super(StateSpace, self).__floordiv__(*args, **kwargs).__finalize__(self)



    def __truediv__(self, *args, **kwargs):
        return super(StateSpace, self).__truediv__(*args, **kwargs).__finalize__(self)



    def __mod__(self, *args, **kwargs):
        return super(StateSpace, self).__mod__(*args, **kwargs).__finalize__(self)



    def __pow__(self, *args, **kwargs):
        return super(StateSpace, self).__pow__(*args, **kwargs).__finalize__(self)



    def __and__(self, *args, **kwargs):
        return super(StateSpace, self).__and__(*args, **kwargs).__finalize__(self)



    def __xor__(self, *args, **kwargs):
        return super(StateSpace, self).__xor__(*args, **kwargs).__finalize__(self)



    def __or__(self, *args, **kwargs):
        return super(StateSpace, self).__or__(*args, **kwargs).__finalize__(self)



    def __iadd__(self, *args, **kwargs):
        return super(StateSpace, self).__iadd__(*args, **kwargs).__finalize__(self)



    def __isub__(self, *args, **kwargs):
        return super(StateSpace, self).__isub__(*args, **kwargs).__finalize__(self)



    def __imul__(self, *args, **kwargs):
        return super(StateSpace, self).__imul__(*args, **kwargs).__finalize__(self)



    def __idiv__(self, *args, **kwargs):
        return super(StateSpace, self).__idiv__(*args, **kwargs).__finalize__(self)



    def __ifloordiv__(self, *args, **kwargs):
        return super(StateSpace, self).__ifloordiv__(*args, **kwargs).__finalize__(self)



    def __imod__(self, *args, **kwargs):
        return super(StateSpace, self).__imod__(*args, **kwargs).__finalize__(self)



    def __ilshift__(self, *args, **kwargs):
        return super(StateSpace, self).__ilshift__(*args, **kwargs).__finalize__(self)



    def __irshift__(self, *args, **kwargs):
        return super(StateSpace, self).__irshift__(*args, **kwargs).__finalize__(self)



    def __iand__(self, *args, **kwargs):
        return super(StateSpace, self).__iand__(*args, **kwargs).__finalize__(self)



    def __ixor__(self, *args, **kwargs):
        return super(StateSpace, self).__ixor__(*args, **kwargs).__finalize__(self)



    def __ior__(self, *args, **kwargs):
        return super(StateSpace, self).__ior__(*args, **kwargs).__finalize__(self)



    def __neg__(self, *args, **kwargs):
        return super(StateSpace, self).__neg__(*args, **kwargs).__finalize__(self)



    def __pos__(self, *args, **kwargs):
        return super(StateSpace, self).__pos__(*args, **kwargs).__finalize__(self)



    def __abs__(self, *args, **kwargs):
        return super(StateSpace, self).__abs__(*args, **kwargs).__finalize__(self)



    def __invert__(self, *args, **kwargs):
        return super(StateSpace, self).__invert__(*args, **kwargs).__finalize__(self)



    def __complex__(self, *args, **kwargs):
        return super(StateSpace, self).__complex__(*args, **kwargs).__finalize__(self)



    def __int__(self, *args, **kwargs):
        return super(StateSpace, self).__int__(*args, **kwargs).__finalize__(self)



    def __long__(self, *args, **kwargs):
        return super(StateSpace, self).__long__(*args, **kwargs).__finalize__(self)



    def __float__(self, *args, **kwargs):
        return super(StateSpace, self).__float__(*args, **kwargs).__finalize__(self)



    def __oct__(self, *args, **kwargs):
        return super(StateSpace, self).__oct__(*args, **kwargs).__finalize__(self)



    def __hex__(self, *args, **kwargs):
        return super(StateSpace, self).__hex__(*args, **kwargs).__finalize__(self)



    def __lt__(self, *args, **kwargs):
        return super(StateSpace, self).__lt__(*args, **kwargs).__finalize__(self)



    def __le__(self, *args, **kwargs):
        return super(StateSpace, self).__le__(*args, **kwargs).__finalize__(self)



    def __eq__(self, *args, **kwargs):
        return super(StateSpace, self).__eq__(*args, **kwargs).__finalize__(self)



    def __ne__(self, *args, **kwargs):
        return super(StateSpace, self).__ne__(*args, **kwargs).__finalize__(self)



    def __ge__(self, *args, **kwargs):
        return super(StateSpace, self).__ge__(*args, **kwargs).__finalize__(self)



    def __gt__(self, *args, **kwargs):
        return super(StateSpace, self).__gt__(*args, **kwargs).__finalize__(self)