


########    IMPORTS    ########



import math
import numpy as np
import pandas as pd
import joblib as jl
import itertools as it
import scipy.stats as st
import scipy.linalg as sl
import warnings
from .regression import LinearRegression



######## CONSTANTS    ########



eps = np.finfo(float).eps   #smallest float, used to avoid divide-by-zero errors



########    METHODS    ########



def IQR(x, axis=0):
    """
    get the interquartile range of x, that is:
                
            IQR = 75 percentile - 25 percentile

    Input:
        x:      (ndarray)
                a numpy ndarray

        axis:   (int)
                the axis of x to be used for IQR calculation.

    Ouput:
        iqr:    (ndarray)
                a 1D array with the IQR of all the dimensions in x accoding to axis.
    """
    
    return np.array(np.quantile(x, 0.75, axis=axis) - np.quantile(x, 0.25, axis=axis)).flatten()



def split_data(source, data, groups):
    """
    return as arrays the data and groups columns contained in source.

    Input:
        source:     (pandas.DataFrame)
                    the dataframe containing the data

        data:       (str, 1D array or list)
                    The names of the source columns containing the data to be investigated.

        groups:     (str, None)
                    The name of the source column containing the grouping variable.

    Output:
        df_data:    (nD array)
                    the array containing the data

        df_groups:  (nD array)
                    the array containing the groups
    """
    # exclude NaNs
    S = source.loc[~source.isna().any(1)].copy()

    # get the data
    df_data = S[data].values.astype(np.float)
    
    # get the groups
    if groups is None:
        df_groups = np.tile(0, df_data.shape[0]).reshape((S.shape[0], len(np.array([groups])))).astype(str)
    else:
        df_groups = S[groups].values.reshape((S.shape[0], len(np.array([groups])))).astype(str)
    
    # split the data by group
    data_out = []
    groups_out = []
    for group in np.unique(df_groups, axis=0):
        
        # get the index
        ix = [np.all([group[i] == df_groups[j, i] for i in np.arange(df_groups.shape[1])])
              for j in np.arange(df_groups.shape[0])]
        ix = np.argwhere(ix).flatten()

        # append the output
        data_out += [df_data[ix].reshape((len(ix), len(np.array([data]).flatten())))]
        groups_out += [df_groups[ix].reshape((len(ix), len(np.array([groups]).flatten())))]

    # return both
    return data_out, groups_out



########    GENERAL CLASSES    ########



class EffectSize():



    def __init__(self, value, name, **kwargs):
        """
        A General class object containing the output of a generic EffectSize.

        Input:
            value:      (float)
                        the value describing the test

            name:       (str)
                        the label describing the type of test.
            
            kwargs:     (any)
                        any additional argument to be stored.
        """

        # add the entered data
        self.name = str(name)
        self.value = np.squeeze([value])
        
        # add all the other parameters
        for i in kwargs:
            setattr(self, i, np.squeeze([kwargs[i]]))

    
    
    def to_df(self):
        """
        Create a column DataFrame containing all the outcomes of the EffectSize object.
        """
        return pd.DataFrame([getattr(self, i) for i in self.attributes()], index=self.attributes(), columns=[0])



    def copy(self):
        """
        create a copy of the current Test object.
        """
        return EffectSize(**{i: getattr(self, i) for i in self.attributes()})



    def attributes(self):
        """
        get the list of attributes stored into this object.
        """
        return [i for i in self.__dict__]



    def __repr__(self):
        return self.__str__()



    def __str__(self):
        return "\n".join([" = ".join([i, str(getattr(self, i))]) for i in self.attributes()])



class Test(EffectSize):



    def __init__(self, value, name, p, alpha, crit, **kwargs):
        """
        A General class object containing the output for the majority of the statistical 0D tests.

        Input:
            value:      (float)
                        the value describing the test

            crit:       (float)
                        the test critical value corresponding to the given alpha value.

            alpha:      (float)
                        the level of significance

            p:          (float)
                        the p-value of the test.

            name:       (str)
                        the label describing the type of test.
            
            kwargs:     (any)
                        any additional argument to be stored.
        """

        # add the entered data
        self.name = str(name)
        self.value = np.array([value]).flatten()[0]
        self.crit = np.array([crit]).flatten()[0]
        self.alpha = np.array([alpha]).flatten()[0]
        self.p = np.array([p]).flatten()[0]
        
        # add all the other parameters
        for i in kwargs:
            setattr(self, i, kwargs[i])

    
    
    def copy(self):
        """
        create a copy of the current Test object.
        """
        return Test(**{i: getattr(self, i) for i in self.attributes()})



class T2(Test):



    def __init__(self, source, data, groups=None, mu=0, paired=True, alpha=0.05):
        """
        Perform a Hotelling T-squared test.

        Input:
            source:     (pandas.DataFrame)
                        the dataframe containing the data

            data:       (str, 1D array or list)
                        The names of the source columns containing the data to be investigated.

            groups:     (str, None)
                        The name of the source column containing the grouping variable.

            mu:         (float or 1D array)
                        The mean value to be used for comparisons.
                        If a float value is given, it is used as mean value for all dimensions.
                        If a 1D array is provided, the length of the array must be equal the number of dimensions of
                        data. Each provided value will be used as mean value for that specific dimension.
            
            paired:     (bool)
                        If both A and B are provided, paired=True will return a paired Hotelling test. Otherwise,
                        a two-sample test is performed. If B is None, paired is ignored.

            alpha:      (float)
                        the level of significance

        Output:
            A Test instance containing the output of the test.

        Note:
            Since the T-squared distribution is not directly available, p values are approximated using an F 
            distribution with corrected degrees of freedom.
        """

        # get the data
        df_data = split_data(source, data, groups)[0]
        
        # ensure mu is a float
        mu = float(mu)

        # check the factors
        M = len(df_data)
        assert M <= 2, "Maximum 2 factors are expected. " + str(M) + " have been found."

        # dimensions
        na, p = df_data[0].shape

        # ONE SAMPLE TEST
        if M == 1:
            t2 = self.__onesample__(df_data[0], mu)
            name = "Hotelling one sample"
            nb = None
        else:
            if paired:
                t2 = self.__onesample__(df_data[1] - df_data[0], mu)
                name = "Hotelling paired"
                nb = None
            else:
                nb = df_data[1].shape[0]
                t2 = self.__twosamples__(df_data[0], df_data[1])
                name = "Hotelling two samples"

        # get the corresponding F value
        correction = (na - p + (0 if nb is None else (nb - 1))) / (p * (na - 1 + (0 if nb is None else (nb - 1))))
        f = correction * t2

        # get the p value
        df_f = (p, na - p + (0 if nb is None else (nb - 1)))
        v = st.f.sf(f, *df_f)

        # get the critical T2 test
        f_crit = st.f.isf(alpha, *df_f)
        t2_crit = f_crit / correction
        
        # get the T2 df
        df = (p, na - 1 + (0 if nb is None else (nb - 1)))

        # create the test
        super(T2, self).__init__(value=t2, df=df, crit=t2_crit, alpha=alpha, p=v, name=name)



    def __onesample__(self, x, m=0):
        """
        return the T2 of the data (x) given the subtracted mean (m)

        Input:
             
            x:  (2D array)
                The data to be tested where each row is a sample and each column a variable.

            m:  (float, 1D array)
                The mean value to be used as H0. If float, the same value is used on all dimensions. If 1D array,
                it must have length equal to x.shape[1] and each value will be used as H0 on the the corresponding
                column of x.

        Output:
            T2: (float)
                The Hotelling T-squared test value
        """
        
        # exclude NaNs
        X = x[~np.isnan(x).any(axis=1), :]
        
        # mean vector
        a = np.mean(X - m, axis=0)

        # sample covariance
        K = (np.cov(X.T) + eps)

        # number of observations
        n = X.shape[0]

        # test value
        return float(a.dot(sl.inv(K)).dot(a.T) * n)



    def __twosamples__(self, a, b):
        """
        return the T2 of the data

        Input:
               
            a:  (2D array)
                The data to be tested where each row is a sample and each column a variable.

            b:  (2D array)
                The second dataset to be compared.

        Output:
            T2: (float)
                The Hotelling T-squared test value
        """
        
        # exclude NaNs
        A = a[~np.isnan(a).any(axis=1)]
        B = b[~np.isnan(b).any(axis=1)]
        
        # shapes
        na, pa = A.shape
        nb, pb = B.shape

        # sample mean
        aa = np.mean(A, 0)
        ab = np.mean(B, 0)

        # pooled covariance
        Ka = np.cov(A, rowvar=False)
        Kb = np.cov(B, rowvar=False)
        Kp = ((na - 1) * Ka + (nb - 1) * Kb) / (na + nb - 2)

        # test
        try:
            return float((na * nb) / (na + nb) * (ab - aa).dot(sl.inv(Kp)).dot((ab - aa).T))
        except Exception:
            return float((na * nb) / (na + nb) * (ab - aa).dot(sl.pinv(Kp, hermitian=True)).dot((ab - aa).T))



class T(Test):



    def __init__(self, source, data, groups=None, mu=0, paired=True, alpha=0.05, two_tailed=True,
                 verbose=True):
        """
        Perform a T test.

        Input:
            source:     (pandas.DataFrame)
                        the dataframe containing the data

            data:       (str, 1D array or list)
                        The names of the source columns containing the data to be investigated.

            groups:     (str, None)
                        The name of the source column containing the grouping variable.

            mu:         (float)
                        The mean value to be used for comparisons.
            
            paired:     (bool)
                        If both A and B are provided, paired=True will return a paired T test.
                        Otherwise, a two-sample test is performed. If B is None, paired is ignored.

            alpha:      (float)
                        the level of significance

            two-tailed: (bool)
                        Should the p-value be calculated from a one or two-tailed distribution?

            verbose:    (bool)
                        Should warning messages be thrown?
        
        Output:
            A Test instance containing the output of the test.
        """

        # get the data
        df_data = split_data(source, data, groups)[0]
        
        # ensure mu is a float
        mu = float(mu)

        # check the factors
        M = len(df_data)
        assert M <= 2, "Maximum 2 factors are expected. " + str(M) + " have been found."

        # dimensions
        na = df_data[0].shape[0]

        # ONE SAMPLE T
        if M == 1:
            t = np.mean(df_data[0] - mu) * np.sqrt(na) / np.std(df_data)
            name = "T one sample"
            df = na - 1
        else:
            
            # PAIRED T
            if paired:
                Z = df_data[0] - df_data[1]
                nz = Z.shape[0]
                t = np.mean(Z - mu) / (np.std(Z) / np.sqrt(nz))
                name = "T paired"
                df = nz - 1
            else:
                nb = df_data[1].shape[0]
                sa = np.var(df_data[0])
                sb = np.var(df_data[1]) 

                # WELCH TEST (in case of unequal variance)
                B = BrownForsythe(source, data, groups, alpha, two_tailed)
                if B.p < alpha:
                    if verbose:
                        warnings.warn("groups have unequal variance. Welch's T test is used.")
                    name = "Welch's T"
                    df = (sa / na + sb / nb) ** 2
                    df /= ((((sa / na) ** 2) / (na - 1)) + (((sb / nb) ** 2) / (nb - 1)))
                    Sp = np.sqrt(sa / na + sb / nb)
                
                # TWO SAMPLES T
                else:
                    name = "T two samples"
                    df = na + nb - 2
                    Sp = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
                    Sp *= np.sqrt(1 / na + 1 / nb)
                
                # T value
                t = (np.mean(df_data[0]) - np.mean(df_data[1])) / Sp
                
        # get the critical T test
        t_crit = st.t.isf(alpha * (0.5 if two_tailed else 1), df)
        
        # adjust the test value in case of a two-tailed test
        if two_tailed:
            t = abs(t)

        # get the p value
        v = st.t.sf(t, df)
        v = np.min([1, 2 * v]) if two_tailed else v

        # create the test
        super(T, self).__init__(value=t, df=df, crit=t_crit, alpha=alpha, two_tailed=two_tailed,
                                p=v, name=name)



class F(Test):



    def __init__(self, SS_num, DF_num, SS_den, DF_den, eps=1, alpha=0.05, two_tailed=True):
        """
        Calculate an F test.

        Input:
            SS_num:     (float)
                        the numerator sum of squares.

            DF_num:     (float)
                        the numerator degrees of freedom.

            SS_den:     (float)
                        the denominator sum of squares.

            DF_den:     (float)
                        the denominator degrees of freedom.
            
            eps:        (float)
                        the epsilon for correcting the degrees of freedom
                        for sphericity. It must be a positive value, such as the
                        Greenhouse-Geisser epsilon, typically within the (0, 1] range.

            alpha:      (float)
                        the level of significance.

            two-tailed: (bool)
                        Should the p-value be calculated from a one or two-tailed distribution?
        
        Output:
            A Test instance containing the output of the test.
        """

        # regressors
        MS_num = SS_num / DF_num
        MS_den = SS_den / DF_den
        
        # get the F value and its degrees of freedom
        f = MS_num / MS_den
        df = (DF_num * eps, DF_den * eps)

        # get the critical F value
        f_crit = st.f.isf(alpha * (0.5 if two_tailed else 1), *df)
                
        # get the p value
        v = st.f.sf(f, *df)

        # create the test
        super(F, self).__init__(value=f, df=df, crit=f_crit, alpha=alpha, two_tailed=two_tailed,
                                p=v, name="F test")
        
        # add some additional parameters
        self.SS_num = SS_num
        self.DF_num = DF_num
        self.MS_num = MS_num
        self.SS_den = SS_den
        self.DF_den = DF_den
        self.MS_den = MS_den
        self.eps = eps



class BrownForsythe(F):



    def __init__(self, source, dv, iv, alpha=0.05, two_tailed=True):
        """
        tests the null hypothesis that the population variances are equal.

        Input:
            source:     (pandas.DataFrame)
                        the dataframe where all variables are stored.

            dv:         (str)
                        the name of the column defining the dependent variable of the model.

            iv:         (list)
                        the list of column names containing the indipendent variables.

            alpha:      (float)
                        the level of significance.

            two_tailed: (bool)
                        should the p-value be calculated from a two-tailed distribution?

        Output:
            A Test instance object containing the outcomes of the test.
        
        Note:
            All factors combinations are assumed to rely on between subjects factors.
        """

        # check the variables
        assert isinstance(source, pd.DataFrame), "source must be a pandas.DataFrame instance."
        assert isinstance(iv, list), "'iv' must be a list instance."
        assert isinstance(dv, list), "'dv' must be a list instance."
        assert len(dv) >= 1, "'dv' must be non-empty."
        for i in iv + dv:
            assert np.any([i == j for j in source]), "{} not found in 'source'.".format(i)
        
        # split the data into unique groups and obtain the residuals against the median
        U = np.unique(source[iv].values.astype(str), axis=0)
        Z = {}
        N = 0
        p = 0
        for j, i in enumerate(U):
            y = source.loc[source[iv].isin(i).all(1)][dv].values.flatten()
            z = abs(y - np.median(y))
            Z[j] = {'zj': np.mean(z), 'nj': len(z), 'zij': z}
            N += len(z)
            p += 1
        Z_avg = np.mean([Z[j]['zj'] for j in Z])

        # get the MS
        MS_num = (N - p) * np.sum([Z[j]['nj'] * ((Z[j]['zj'] - Z_avg) ** 2) for j in Z])
        MS_den = (p - 1) * np.sum([np.sum((Z[j]['zij'] - Z[j]['zj']) ** 2) for j in Z])
                        
        # get the df
        DF_num = p - 1
        DF_den = N - p

        # get the SS
        SS_num = MS_num * DF_num
        SS_den = MS_den * DF_den

        # store the outcomes
        super(BrownForsythe, self).__init__(SS_num, DF_num, SS_den, DF_den, eps=1, alpha=0.05,
                                            two_tailed=two_tailed)
        self.name = "Brown-Forsythe"



class PermutationTest(Test):



    def __init__(self, source, data, groups, test=T, k=1000, alpha=0.05, two_tailed=True, verbose=True,
                 test_kwargs=None):
        """
        Perform a non-parametric permutation test based on the test required.

        Input:
            source:     (pandas.DataFrame)
                        the dataframe containing the data

            data:       (str, 1D array or list)
                        The names of the source columns containing the data to be investigated.

            groups:     (str, 1D array, list, None)
                        The names of the source columns containing the grouping variables.
            
            test:       (Test)
                        a Test instance used as statistical test after each permutation.
            
            k:          (int)
                        the number of permutations to be computed. If n_perm is higher than the maximum number of
                        permutations effectively available for the given data a warning is thrown.

            alpha:      (float)
                        the level of significance

            two_tailed: (float)
                        should the p-value be calculated from a two-tailed distribution?
            
            verbose:    (bool)
                        Should warning messages be thrown?

            test_kwargs: named parameters to be passed directly to test.
        
        Output:
            A Test object containing the outcomes of the test.
        """

        # get the total number of possible permutations
        N = math.factorial(source.shape[0])
        if N < k and verbose:
            txt = " ".join(["The total number of possible permutations are", str(N), "while n_permutations =", str(k)])
            warnings.warn(txt)

        # build the pdf
        # pdf = np.squeeze([self.__ptest__(source, data, groups, test, test_kwargs) for i in np.arange(k)])  # debug
        pdf = np.squeeze(jl.Parallel(n_jobs=-1, prefer="threads")
                         (jl.delayed(self.__ptest__)(source, data, groups, test, test_kwargs) for i in np.arange(k)))

        # get the test output and standardize test output
        t0 = test(source, data, groups, **test_kwargs)

        # get the critical value
        crit = np.quantile(pdf, 1 - alpha)

        # handle the two_tailed case
        if two_tailed:
            p = (np.sum(abs(pdf - np.mean(pdf)) > abs(t0.value - np.mean(pdf))) + 1) / (k + 1)
        else:
            p = (np.sum(pdf > t0.value) + 1) / (k + 1)
        
        # return the output of the test
        super(PermutationTest, self).__init__(value=t0.value, permutations=k, crit=crit, alpha=alpha, p=p,
                                              two_tailed=two_tailed, name="Permuted " + t0.name)


    def __ptest__(self, source, data, groups, test=T, test_kwargs=None):
        """
        apply the test after having permuted the data.

        Input:
            source:     (pandas.DataFrame)
                        the dataframe containing the data

            data:       (str, 1D array or list)
                        The names of the source columns containing the data to be investigated.

            groups:     (str, 1D array, list, None)
                        The names of the source columns containing the grouping variables.
            
            test:       (Test)
                        a Test instance used as statistical test after each permutation.

            test_kwargs: named parameters to be passed directly to test.
            
        Output:
            A Test object containing the outcomes of the test.
        """

        # get a copy of source with the order of the groups permuted
        np.random.seed()
        ix = source.index.to_numpy()
        S = source[np.squeeze(np.append(data, groups))].copy()
        S.loc[ix, groups] = S.loc[np.random.permutation(ix), groups].values

        # return the (max) value of the test resulting from the shuffled data
        return np.max(test(source=S, data=data, groups=groups, **test_kwargs).value)



class JohnNagaoSugiura(Test):



    def __init__(self, effect, alpha, two_tailed):
        """
        Perform the John, Nagao and Sugiura test to asses the sphericity of an Effect.

        Input:

            effect:     (AnovaEffect)
                        an object of class AnovaEffect
            
            alpha:      (float)
                        the level of significance.

            two_tailed: (bool)
                        should the p-value be calculated from a two-tailed distribution?
        """

        if effect.isBetween:
            return None
        
        # get the (positive) eigenvalues from the errors
        E = effect.eigenvalues(effect.SSPE)
        E = E.values.flatten()[E.values.flatten() > 0]
        
        # get the test statistic
        p = effect.DFn
        n = int(effect.DFd / p)
        W = np.sum(E) ** 2 / np.sum(E ** 2)
        Q = n / 2 * (p - 1) ** 2 * (W - 1 / (p - 1))
        df = (p + 1) * p / 2 - 1
        
        # obtain critical and p value
        Q_crit = st.chi2.isf(alpha * (0.5 if two_tailed else 1), df)
                
        # get the p value
        v = st.chi2.sf(Q, df)

        # create the test
        super(JohnNagaoSugiura, self).__init__(value=Q, df=df, crit=Q_crit, alpha=alpha,
                                               two_tailed=two_tailed, p=v,
                                               name="John, Nagao, Sugiura test")



class Cohen_d(EffectSize):



    def __init__(self, source, data, groups=None):
        """
        Compute the Cohen's d effect size between 2 groups.

        Input:
            source:     (pandas.DataFrame)
                        the dataframe containing the data

            data:       (str)
                        The name of the source column containing the data to be investigated.

            groups:     (str)
                        The name of the source column containing the grouping variable.
        
        Output:
            A Test instance containing the output of the test.

        Note:
            This test does not provide p, alpha or crit values. However its test value can be interpreted according to the
            following table:

                    Effect size     d       Reference
                    Very small      0.01	Sawilowsky, 2009
                    Small	        0.20	Cohen, 1988
                    Medium	        0.50	Cohen, 1988
                    Large       	0.80	Cohen, 1988
                    Very large  	1.20	Sawilowsky, 2009
                    Huge	        2.0 	Sawilowsky, 2009

        References:
            Sawilowsky, S (2009). "New effect size rules of thumb". Journal of Modern Applied Statistical Methods.
                8(2): 467–474. doi:10.22237/jmasm/1257035100.
        """

        # get the data
        df_data, df_groups = split_data(source, data, groups)
                
        # check the factors and data
        assert df_data.ndim == 1, "data must be a single variable."
        M = len(np.unique(df_groups))
        assert M == 2, "2 factors are expected. " + str(M) + " have been found."

        # split data by group
        F1 = np.argwhere(df_groups == np.unique(df_groups)[0]).flatten()
        F2 = np.argwhere(df_groups == np.unique(df_groups)[1]).flatten()
        A = df_data[F1]
        B = df_data[F2]

        # get size and variance
        Na = A.shape[0]
        Nb = B.shape[0]
        Sa = np.var(A)
        Sb = np.var(B)

        # get the pooled SD
        Sp = np.sqrt(((Na - 1) * Sa + (Nb - 1) * Sb) / (Na + Nb - 2))

        # get the d
        d = (np.mean(B) - np.mean(A)) / Sp

        # return the test
        super(Cohen_d, self).__init__(d, "Cohen's d")



class TotalEtaSquare(EffectSize):



    def __init__(self, num_SS=None, tot_SS=None):
        """
        Compute the partial eta squared effect size.

        Input:
            num_SS: (float, None)
                    The numerator sum of squares.
            
            tot_SS: (float, None)
                    The denominator sum of squares.
        """
       
        # create the instance
        super(TotalEtaSquare, self).__init__(num_SS / tot_SS, "Total EtaSq")



class PartialEtaSquare(EffectSize):



    def __init__(self, num_SS=None, den_SS=None):
        """
        Compute the partial eta squared effect size.

        Input:
            F_test: (F)
                    An F test object. If passed, num_SS and den_SS are ignored.
            
            num_SS: (float, None)
                    The numerator sum of squares. Ignored if F is not None.
            
            den_SS: (float, None)
                    The denominator sum of squares. Ignored if F is not None.
        """

        # from F
        if F_test is not None:
            assert isinstance(F_test, F), "F_test must be an instance of the F class."
            pes = F_test.num_SS / (F_test.num_SS + F_test.den_SS)
        
        # from SSs
        else:
            pes = num_SS / (num_SS + den_SS)
        
        # create the instance
        super(PartialEtaSquare, self).__init__(pes, "Partial EtaSq")



class GeneralizedEtaSquare(EffectSize):



    def __init__(self, num_SS=None, *args, **kwargs):
        """
        Compute the generalized eta square effect size.

        Input:            
            num_SS: (float, None)
                    The numerator sum of squares. Ignored if F is not None.
            
            args:   (float)
                    The list of sum of squares defining the errors.
        """

        # create the instance
        err_SS = np.sum([i for i in args] + [kwargs[j] for j in kwargs])
        super(GeneralizedEtaSquare, self).__init__(num_SS / (num_SS + err_SS), "Generalized EtaSq")



class AnovaEffect():



    def __init__(self, SSt, DFt, BW, design, hypothesis, isBetween, label):
        """
        basically a container for the data corresponding to each effect of an Anova object.

        Input:

            SSt:            (float)
                            Sum of Squares of the model.

            DFt:            (float)
                            Sum of Squares of the model.

            BW:             (pyomech.LinearRegression)
                            the within ~ between regression object

            design:         (pandas.DataFrame)
                            the matrix containing the effect design contrasts.
            
            hypothesis:     (pandas.DataFrame)
                            the hypothesis matrix of the effect.
            
            isBetween:      (bool)
                            is this effect a between-subjects effect?
            
            label:          (str)
                            the string label defining the effect.
        """
        
        # add the entries
        self.SSt = SSt
        self.DFt = DFt
        self.isBetween = isBetween
        self.label = label
        
        # dimensions
        n, p = design.shape

        # inverted design cross product
        PPi = sl.inv(design.T.dot(design))
        self.__PPi__ = pd.DataFrame(PPi, index=design.columns, columns=design.columns)

        # effect term
        V = hypothesis.dot(BW.cov_unscaled()).dot(hypothesis.T)           # covariance
        Vi = pd.DataFrame(sl.inv(V), index=V.columns, columns=V.columns)  # inverted covariance
        B = hypothesis.dot(BW.coefs.dot(design))                          # coefs
        self.SSP = B.T.dot(Vi).dot(B)
        self.SSn = np.sum(np.diag(self.SSP.dot(self.__PPi__)))           
        self.DFn = p

        # error term
        self.SSPE = design.T.dot(BW.SSPE()).dot(design)
        self.SSd = np.sum(np.diag(self.SSPE.dot(self.__PPi__)))
        self.DFd = n * p
        


    def F_test(self, eps=1, alpha=0.05, two_tailed=True):
        """
        return the F test calculated with the required Epsilon.

        Input:
            eps:    (float)
                    a value in the (0, 1] range defining the sphericity correction
                    to be applied to the corresponding degrees of freedom.
            
            alpha:      (float)
                        the level of significance.

            two_tailed: (bool)
                        should the p-value be calculated from a two-tailed distribution?
        
        Output:
            f:      (pandas.DataFrame)
                    a dataframe containing the output of the F test.
        """

        # F test
        f = F(self.SSn, self.DFn, self.SSd, self.DFd, eps, alpha, two_tailed)

        # index
        cx = [np.tile("F test", 6), ['SSn', 'SSd', 'Statistic', 'DF', 'Critical', 'P']]
        C = pd.MultiIndex.from_arrays(np.atleast_2d(cx))
        
        # dataframe
        ln = [f.SS_num, f.SS_den, f.value, (f.DF_num, f.DF_den), f.crit, f.p]
        return pd.DataFrame(ln, index=C, columns=[self.label]).T



    def sphericity_test(self, alpha=0.05, two_tailed=True):
        """
        return the output of a John, Nagao, Sugiura test to evaluate if the effect of
        sphericity. In addition, return Greenhouse-Geisser and Huynh-Feldt corrections.
        Finally provide the suggested p-values and df according to the output of the test.

        Input:            
            alpha:      (float)
                        the level of significance.

            two_tailed: (bool)
                        should the p-value be calculated from a two-tailed distribution?
        
        Output:
            v:          (pandas.DataFrame)
                        a dataframe containing the output of the analysis.
        """
       
        #* JOHN-NAGAO-SUGIURA TEST
        
        # index
        cx = [np.tile("JNS test", 4), ['Statistic', 'DF', "Crit", 'P']]
        C = pd.MultiIndex.from_arrays(np.atleast_2d(cx))
        
        # outcomes
        if self.label == "Intercept" or self.isBetween:
            ln = np.tile(None, 4)
            jns = None
        else:
            jns = JohnNagaoSugiura(self, alpha, two_tailed)
            ln = [jns.value, jns.df, jns.crit, jns.p]
        
        # dataframe
        test = pd.DataFrame(ln, index=C, columns=[self.label]).T


        #* GREENHOUSE-GEISSER SPHERICITY CORRECTION
        
        # index
        cx = [np.tile("Greenhouse-Geisser", 4), ['Epsilon', 'DF', "Crit", 'P']]
        C = pd.MultiIndex.from_arrays(np.atleast_2d(cx))
        
        # outcomes
        if self.label == "Intercept" or self.isBetween:
            ln = np.tile(None, 4)
        else:
            gg_eps = self.epsilon_GG()
            gg_f = F(self.SSn, self.DFn, self.SSd, self.DFd, gg_eps, alpha, two_tailed)
            ln = [gg_eps, gg_f.df, gg_f.crit, gg_f.p]
        
        # dataframe
        gg = pd.DataFrame(ln, index=C, columns=[self.label]).T
        
        
        #* HUYNH-FELDT SPHERICITY CORRECTION
        
        # index
        cx = [np.tile("Huynh-Feldt", 4), ['Epsilon', 'DF', "Crit", 'P']]
        C = pd.MultiIndex.from_arrays(np.atleast_2d(cx))
        
        # outcomes
        if self.label == "Intercept" or self.isBetween:
            ln = np.tile(None, 4)
        else:
            hf_eps = self.epsilon_HF()
            hf_f = F(self.SSn, self.DFn, self.SSd, self.DFd, hf_eps, alpha, two_tailed)
            ln = [hf_eps, hf_f.df, hf_f.crit, hf_f.p]
        
        # dataframe
        hf = pd.DataFrame(ln, index=C, columns=[self.label]).T
        
        
        #* SUGGESTED CORRECTION

        # index
        cx = [np.tile("Suggested correction", 1), ['']]
        C = pd.MultiIndex.from_arrays(np.atleast_2d(cx))
        
        # outcomes
        if jns is None:
            ln = np.tile(None, 1)
        else:
            if jns.p < alpha:
                if gg_eps <= 0.7:
                    ln = ['Greenhouse-Geisser']
                else:
                    ln = ['Huynh-Feldt']
            else:
                ln = ['F test']
                    
        # dataframe
        opt = pd.DataFrame(ln, index=C, columns=[self.label]).T


        #* MERGE OUTCOMES AND RETURN
        return pd.concat([test, gg, hf, opt], axis=1)
            


    def variance_test(self, alpha=0.05, two_tailed=True):
        """
        return the output of a Brown-Forsythe test to evaluate the equality of variance of
        the effect.
        """

        # index
        cx = [np.tile('Brown-Forsythe', 4), ["Statistic", 'DF', "Crit", 'P']]
        C = pd.MultiIndex.from_arrays(cx)
        
        # values
        if not self.isBetween:
            ln = np.tile(None, 4)
        else:
            BF = BrownForsythe(self.source, self.DV, self.IV, alpha, two_tailed)
            ln = [BF.value, (BF.DF_num, BF.DF_den), BF.crit, BF.p]
        
        # return the dataframe
        return pd.DataFrame(ln, index=C, columns=[self.label]).T



    def eigenvalues(self, SS_matrix):
        """
        return the eigenvalues associated to the Effect (SSP) or Error (SSPE).

        Input:
            SS_matrix:  (pandas.DataFrame)
                        the dataframe containing the sum of square product (error) matrix.
        
        Output:
            eig:        (pandas.DataFrame)
                        a dataframe containing the eigenvalue associated to each element of the
                        effect.
        """
        eig = np.real(sl.eigvals(SS_matrix.dot(self.__PPi__)))
        return pd.DataFrame(eig, index=self.__PPi__.columns).T



    def epsilon_GG(self):
        """
        return the Greenhouse-Geisser epsilon for sphericity correction.
        """
        if self.isBetween:
            return 1
        else:
            
            # get the positive eigenvalues from SSPE
            E = self.eigenvalues(self.SSPE).values.flatten()
            E = E[E > 0]
            
            # return the Greenhouse-Geisser epsilon
            p = self.DFn
            return ((np.sum(E) / p) ** 2) / (np.sum(E ** 2) / p)



    def epsilon_HF(self):
        """
        return the Huynh-Feldt epsilon for sphericity correction.
        """
        if self.isBetween:
            return 1
        else:
            
            # get the positive eigenvalues from SSPE
            E = self.eigenvalues(self.SSPE).values.flatten()
            E = E[E > 0]
            
            # return the Huynh-Feldt epsilon
            p = self.DFn
            n = int(self.DFd / p)
            GG = self.epsilon_GG()
            return ((n + 1) * p * GG - 2) / (p * (n - p * GG))



    def effect_sizes(self, SS_error):
        """
        Return the total, partial and generalized Eta squares for the effect

        Input:
            SS_error:   (float)
                        the sum of squares of the error terms of the full model
        
        Output:
            ges:        (pandas.DataFrame)
                        the generalized eta square for this effect
        """

        # column index
        cx = [np.tile("Eta square", 3), ['Total', 'Partial', 'Generalized']]
        C = pd.MultiIndex.from_arrays(np.atleast_2d(cx))

        # effects
        ln = self.SSn / np.array([self.SSt, self.SSn + self.SSd, self.SSn + SS_error])
        return pd.DataFrame(ln, columns=[self.label], index=C).T
        


class Anova(LinearRegression):



    def __init__(self, source, dv, iv, subjects=None, alpha=0.05, two_tailed=True):
        """
        Generate an Anova class object.

        Input:
            source:         (pandas.DataFrame)
                            the dataframe where all variables are stored.

            dv:             (str)
                            the name of the column defining the dependent variable of the model.

            iv:             (list)
                            the list of column names containing the indipendent variables.
                                    
            subjects:       (list or None)
                            If none, the factors are treated as "between-only" factors. Conversely,
                            if a list or a 1D ndarray is provided, the values are used to define
                            within-subjects errors.
                            If such value is provided, it is also added to source.

            alpha:          (float)
                            the level of significance.

            two_tailed:     (bool)
                            should the p-value be calculated from a two-tailed distribution?
        """
        
        #* DATA PREPARATION

        # check the variables
        assert isinstance(source, pd.DataFrame), "source must be a pandas.DataFrame instance."
        assert isinstance(iv, list), "'iv' must be a list instance."
        assert isinstance(dv, list), "'dv' must be a list instance."
        assert len(dv) >= 1, "'dv' must be non-empty."
        for i in iv + dv:
            assert np.any([i == j for j in source]), "{} not found in 'source'.".format(i)
        self.DV = dv
        self.IV = iv
        self.alpha = alpha
        self.two_tailed = two_tailed
        
        # check the subjects
        if subjects is not None:
            txt = "{} not found in 'source'.".format(subjects)
            assert np.any([subjects == i for i in source]), txt
            SRC = source.copy()

        else:
            SRC = source.astype(str)
            unique_combs = np.unique(SRC[iv].values, axis=0)
            for i in np.arange(len(unique_combs)):
                cmb = unique_combs[i]
                name = 'S{}'.format(i + 1)
                index = SRC.loc[SRC.isin(cmb).all(1)].index.to_numpy()
                SRC.loc[index, 'SBJ'] = name
            subjects = 'SBJ'
                
        # regroup the source
        GRP = SRC.groupby(iv + [subjects], as_index=False).mean()
        self.source = GRP[iv + dv]
        self.source.index = pd.Index(GRP[subjects].values.flatten())

        # separate between and within factors
        BV = []
        WV = []
        for i in self.source:
            if not np.any([i == j for j in dv]):
                if self.__isBetween__(pd.DataFrame(self.source[i])):
                    BV += [i]
                else:
                    WV += [i]
        self.between = BV
        self.within = WV
        self.covariates = [i for i in self.IV if self.__isCovariate__(pd.DataFrame(self.source[i]))]

        # initialize the object
        X = [self.__contrasts__(self.source[i.split(":")])
             for i in self.__combine__(self.between + self.within)]
        X = pd.concat(X, axis=1)
        Y = pd.DataFrame(self.source[dv])
        super(Anova, self).__init__(Y, X)

        # Total sum of squares and degrees of freedom
        SSt = np.sum((self.Y - self.Y.mean()).values ** 2)
        DFt = self.source.shape[0] - 1


        #* WITHIN <- BETWEEN REGRESSION

        # get the between and within subjects variable combinations
        BV_combs = {'Intercept': ["Intercept"], **self.__combine__(self.between)}
        WV_combs = {'Intercept': ["Intercept"], **self.__combine__(self.within)}

        # create the between-subjects data
        GRP = self.source.copy().reset_index()
        GRP.insert(0, "SBJ", self.source.index.to_numpy())
        if len(self.between) > 0:
            BV_src = GRP.groupby(self.between + ['SBJ'], as_index=False).mean()
            BV_src.index = pd.Index(BV_src['SBJ'].values.flatten())
            X = []
            for i in BV_combs:
                if i != "Intercept":
                    X += [self.__contrasts__(pd.DataFrame(BV_src[i.split(":")]))]
            X = pd.concat(X, axis=1)
        else:
            X = pd.DataFrame(index=GRP.groupby(['SBJ']).mean().index.to_numpy())

        # create the within-subjects data
        if len(self.within) > 0:
            Z = GRP.groupby(self.within + ['SBJ']).mean().unstack(self.within)[self.DV]
            Z.columns = pd.Index([":".join(i[1:]) for i in Z.columns])
            Z.index = X.index
        else:
            Z = pd.DataFrame(GRP[self.DV])
            Z.index=BV_src.index
            Z.columns=["Intercept"]

        # perform the regression
        LM = LinearRegression(Z, X)
        
        # between-effects selection matrix
        Ip = pd.DataFrame(np.eye(LM.coefs.shape[0]), index=LM.coefs.index, columns=LM.coefs.index)

        
        #* EFFECTS CALCULATION

        # iterate the calculation of the stats
        self.effects = {}
        for i, w in enumerate(WV_combs):
            
            # design matrix of the within effects
            # P = self.__design_matrix__(include_between=False)[w]
            if w == "Intercept":
                P = self.__design_matrix__(self.source[self.within], include_intercept=True)
            else:
                P = self.__design_matrix__(self.source[self.within], w, include_intercept=False)
            
            # interaction with between-effects
            for j, b in enumerate(BV_combs):

                # get the label
                label = b if w == "Intercept" else (w if b == "Intercept" else ":".join([b, w]))

                # hypothesis (used to deal with between-within relationship)
                L = Ip[np.concatenate([k.split(":") for k in BV_combs[b]]).flatten()]
                L = pd.DataFrame(L).T
                
                # between
                isBetween = (w == "Intercept") & (b != "Intercept")

                # store the effect
                self.effects[label] = AnovaEffect(SSt, DFt, LM, P, L, isBetween, label)           



    def anova_table(self):
        """
        Return a pandas.DataFrame containing the Anova analysis of the entered model.
        """
        
        # get the total error SS to calculate the generalized effect sizes
        SSe = np.sum([self.effects[e].SSd for e in self.effects])
        
        # create the table
        table = pd.DataFrame()
        for e in self.effects:
            contents = [self.effects[e].variance_test(), self.effects[e].F_test(),
                        self.effects[e].sphericity_test(), self.effects[e].effect_sizes(SSe)]
            table = table.append(pd.concat(contents, axis=1))

        # drop empty columns and return
        return table.dropna(axis=1, how='all')



    def normality_table(self):
        """
        Return a table containing the Shapiro-Wilk test calculated from both the
        residuals and all the model combinations.
        """
        cx = np.atleast_2d([np.tile("Shapiro-Wilk test", 2), ['Statistic', 'P']])
        C = pd.MultiIndex.from_arrays(cx)
        S = st.shapiro(self.residuals().values.flatten())
        return pd.DataFrame(S, index=C, columns=['Residuals']).T



    def emm_table(self):
        """
        Return Estimated Marginal MEANS for each group combination.
        """

        # iterate each effect
        effects = [i for i in self.effects if i != "Intercept"]
        self.emmeans = {e: self.__emmeans__(self.effects[e]) for e in effects}
        check = 1



    def __emmeans__(self, E):
        """
        return the estimated marginal means for an effect given through its linear function and its
        scaled covariance matrix.
        
        Input:
            E:  (AnovaEffect)
                The effect about which the emmeans have to be calculated.

        Output:
            M:  (pandas.DataFrame)
                a dataframe containing the emmeans for the effect.
        """

        # obtain the linear function
        I = np.unique(self.source[E.label.split(":")].values.astype(str), axis=0)
        I = pd.Index([":".join(i) for i in I])
        C = self.coefs.index
        L = pd.DataFrame(np.zeros((len(I), len(C))), index=I, columns=C)
        P = self.__design_matrix__(self.source[E.label.split(":")], E.label, include_intercept=True)
        L.loc[P.index, P.columns] = P.values

        # get the scaled covariance matrix
        V = self.cov_unscaled()
        Ic = self.effects["Intercept"].SSd / self.effects["Intercept"].DFd
        V.loc["Intercept", "Intercept"] *= Ic
        C = pd.Index([i for i in P.columns if i != "Intercept"])
        V.loc[C, C] *= (E.SSd / E.DFd)
        M = L.dot(V) * L

        # get the estimates
        em = L.dot(self.coefs).T
        em.index = pd.Index(['Estimate'])
            
        # obtain the standard errors
        se = pd.DataFrame(M.sum(1)).T.apply(np.sqrt)
        se.index = pd.Index(['Standard error'])
        
        # get the degrees of freedom
        df = pd.DataFrame()
        DFd = pd.DataFrame()
        for e in self.effects:
            D = [pd.DataFrame(M.sum(1))
            D.columns = pd.Index([e])
            df = pd.concat([df, D], axis=1)
            F = D.copy()
            F.loc[F.index] = self.effects[e].DFd
            DFd = pd.concat([DFd, F], axis=1)
        df = pd.DataFrame((df.sum(1) ** 2) / (df ** 2 / DFd).sum(1)).T
        df.index = pd.Index(["DF"])
                            
        # critical T value
        tc = df.copy()
        al = self.alpha * (0.5 if self.two_tailed else 1)
        for t in tc:
            tc.loc[tc.index, t] = st.t.isf(al, tc.loc[tc.index, t])
        tc.index = pd.Index(['T crit'])

        # confidence intervals
        ci_inf = em.copy()
        ci_sup = em.copy()
        for t in ci_inf:
            c = tc.loc[tc.index, t].values * se.loc[se.index, t].values
            ci_inf.loc[ci_inf.index, t] = em.loc[em.index, t].values - c
            ci_sup.loc[ci_sup.index, t] = em.loc[em.index, t].values + c
        ci_inf.index = pd.Index(["{:0.0f}% inf C.I.".format(100 * (1 - self.alpha))])
        ci_sup.index = pd.Index(["{:0.0f}% sup C.I.".format(100 * (1 - self.alpha))])

        # T value
        tv = em.copy() / se.values
        tv.index = pd.Index(['T stat'])

        # get the test p values
        pv = tv.copy()
        for t in tv:
            pv.loc[pv.index, t] = st.t.sf(tv.loc[tv.index, t], df.loc[df.index, t])
        pv.index = pd.Index(['P'])

        # get the effect degrees of freedom
        dfd = df.copy()
        dfd.loc[dfd.index] = E.DFd

        # return the table
        pd.concat([em, se, df, tc, ci_inf, ci_sup, tv, dfd, tc, pv], axis=0).T


    def emm_contrasts_table(self):
        """
        Return pairwise contrasts based on estimated marginal means for each effect.
        """

        # get the covariance matrix scaled by the MS of the errors
        # in addition, obtain the variance of the effect and the error
        # degrees of freedom.
        V = self.cov_unscaled()
        D = pd.DataFrame(index=pd.Index(["DF"]))
        S = pd.DataFrame(index=pd.Index(["DF"]))
        for e in self.effects:
            D.insert(D.shape[1], e, [self.effects[e].DFd])
            MSE = self.effects[e].SSd / self.effects[e].DFd
            idx = self.effects[e].__PPi__.columns
            V.loc[idx, idx] *= MSE
            S.insert(S.shape[1], e, [np.mean(np.mean(V.loc[idx, idx]))])
        
        # get the degrees of freedom
        ddf = (np.sum(S.values) ** 2) / np.sum((S.values ** 2) / D.values)

        # iterate each effect
        effects = [i for i in self.effects if i != "Intercept"]
        self.emmeans = {}
        for e in effects:
            
            # obtain the linear function
            I = np.unique(self.source[e.split(":")].values.astype(str), axis=0)
            I = pd.Index([":".join(i) for i in I])
            C = self.coefs.index
            L = pd.DataFrame(np.zeros((len(I), len(C))), index=I, columns=C)
            P = self.__design_matrix__(self.source[e.split(":")], e, include_intercept=True)
            C = pd.Index([i for i in P.columns if i != "Intercept"])
            L.loc[P.index, P.columns] = P.values
            
            # get the estimates
            em = L.dot(self.coefs).T
            em.index = pd.Index(['Estimate'])
            
            # obtain the standard errors
            se = pd.DataFrame((L.dot(V) * L).sum(1)).T.apply(np.sqrt)
            se.index = pd.Index(['Standard error'])
            
            # TODO get the degrees of freedom
            """
            df = se.copy()
            df.index = pd.Index(["DF"])
            df.loc[df.index] = np.atleast_2d(ddf)
                            
            # critical T value
            tc = df.copy()
            al = self.alpha * (0.5 if self.two_tailed else 1)
            for t in tc:
                tc.loc[tc.index, t] = st.t.isf(al, tc.loc[tc.index, t])
            tc.index = pd.Index(['T crit'])

            # confidence intervals
            ci = em.copy()
            for t in ci:
                c = tc.loc[tc.index, t].values * se.loc[se.index, t].values * np.array([-1, 1])
                ci.loc[ci.index, t] = "{}-{}".format(*(em.loc[em.index, t].values + c).tolist())
            ci.index = pd.Index(["{:0.0f}% C.I.".format(100 * (1 - self.alpha))])

            # create the ref table
            self.emmeans[e] = pd.concat([em, se, tv, df, tc, ci], axis=0).T
            """
            # obtain the contrast matrix
            J = np.atleast_2d(np.unique(self.source[e.split(":")].values.astype(str), axis=0))
            M = pd.DataFrame(index=P.index)
            for i, j in enumerate(J[:-1]):
                for z in J[(i + 1):]:
                    col = " - ".join([":".join(j), ":".join(z)])
                    cmj = pd.DataFrame(np.zeros((P.shape[0], 1)), index=P.index, columns=[col])
                    cmj.loc[":".join(j)] = 1
                    cmj.loc[":".join(z)] = -1
                    M = pd.concat([M, cmj], axis=1)
            
            # get the linear function for the contrasts
            Lc = M.T.dot(L)


    def descriptive(self):
        """
        Return descriptive statistics for each group combination along with T-tests
        ("two-samples" for between-subjects comparisons and "paired" for "within-subjects")
        and Holm-corrected p values.
        """
        return self.__describe__(self.source)



    def copy(self):
        """
        return a copy of the current instance.
        """
        df = self.source
        df.insert(0, 'SBJ', self.source.index.to_numpy())
        return Anova(df, self.DV, self.IV, 'SBJ', self.alpha, self.two_tailed)



    def __combine__(self, x):
        """
        Internal function used to combine labels and return the groups combinations according to the
        available data.

        Input:

            x:  (list)
                a list of str.
        
        Output:

            c:  (dict)
                a dict with each factor combination as key and the corresponding factors as values.
        """
        return {":".join(j): 
                self.__contrasts__(self.source[list(j)]).columns.to_numpy().tolist()
                for i in np.arange(len(x)) + 1 for j in it.combinations(x, i)}



    def __isBetween__(self, df):
        """
        Check whether the current parameter can be considered as a between-subjects variable.

        Input:
            df: (pandas.DataFrame)
                The dataframe representing a variable.
    
        Output:
            B:  (bool)
                True if the variable can be handled as a betwee-subjects variable. False, otherwise.
        """
        if self.__isCovariate__(df):
            return True
        for s in np.unique(df.index.to_numpy()):
            if len(np.unique(df.loc[s].values.astype(str), axis=0)) > 1:
                return False
        return True



    def __isCovariate__(self, df):
        """
        Check if the current variable can be considered as a covariate.

        Input:
            df: (pandas.DataFrame)
                The dataframe representing a variable.
    
        Output:
            C:  (bool)
                True if the variable can be handled as a covariate. False, otherwise.
        """
        
        # get the covariance types
        cov_types = np.array([['float{}'.format(i), 'int{}'.format(i)] for i in [16, 32, 64]])
        cov_types = cov_types.flatten().tolist()
        
        # if any of the columns in source are factors, the resulting variable will be a factor
        return df.select_dtypes(cov_types).shape[1] == df.shape[1]



    def __contrasts__(self, df, type="sum"):
        """
        create a dummy representation of a variable.
        
        Input:
            df:     (pandas.DataFrame)
                    a pandas dataframe where each column represents one component of a
                    linear model variable.

            type:   (str)
                    any of "treat" or "sum". The former will have only 1 and 0.
                    The latter, 1, 0 and -1.
        
        Output:
            D:      (pandas.DataFrame)
                    a dataframe containing dummy variables to represent the type
                    of model.
        """

        # check the type
        types = ['sum', 'treat']
        assert np.any([type == i for i in types]), "'type' must be any of " + str(types)

        # get the groups combinations
        D = {}
        for prm in df.columns:
            X = pd.DataFrame(df[prm])
            X.index = df.index

            # handle covariates
            if self.__isCovariate__(X):
                D[prm] = X

            # handle factors
            else:

                # get the groups combinations
                G = np.unique(X.values.flatten())
                I = [np.argwhere(X.values.flatten() == i).flatten() for i in G]
                    
                # get the dummy columns
                dummy = np.zeros((X.shape[0], len(G) - 1))
                indices = np.arange(1, len(G)) if type == "treat" else np.arange(len(G) - 1)
                cols = [label for label in G[indices]]
                D[prm] = pd.DataFrame(dummy, index=df.index, columns=cols)
                    
                # fill the columns
                for i in indices:
                    D[prm][G[i]].iloc[I[i]] = 1
                if type == "sum":
                    D[prm].iloc[I[-1]] = -1

        # combine the groups
        G = [j for j in it.product(*[D[i].columns.to_numpy() for i in D], repeat=1)]
        V = pd.concat([D[i] for i in D], axis=1)
        F = {":".join(i): np.prod(V[[k for k in i]], axis=1).values.flatten() for i in G}
        I = pd.Index([":".join(i) if len(np.array([i]).flatten()) > 1 else i for i in df.index])
        return pd.DataFrame(F, index=I)



    def __design_matrix__(self, source, effect="", include_intercept=True, type="sum"):
        """
        obtain the design matrix for 'effect' on source.

        Input:
            source:             (pandas.DataFrame)
                                the dataframe containing the combinations to be used as
                                reference (i.e. the rows of the design matrix)

            effect:             (str)
                                the effect of which the design is required
                                (i.e. the columns of the design matrix)

            include_intercept:  (bool)
                                Should the intercept be included in the X matrix?
            
            type:               (str)
                                The type of contrasts to be provided. The options are "sum" or
                                "treat".

        Output:
            K:                  (dict)
                                a dict having each indipendent variable as key which maps a
                                pandas dataframe containing the corresponding design matrix.                               
        """
        
        # check the type
        types = ['sum', 'treat']
        assert np.any([type == i for i in types]), "'type' must be any of {}.".format(str(types))

        # check the include_intercept
        assert include_intercept or not include_intercept, "'include_intercept' must be a boolean."
        
        # check source
        assert isinstance(source, pd.DataFrame), "'source' must be a pandas.DataFrame object."

        # check effect
        assert isinstance(effect, str), "'effect' must be a string."
        cols = [] if effect == "" else effect.split(":")
        for c in cols:
            assert np.any([c == j for j in source.columns]), "{} not found in 'source'.".format(c)

        # get the reference data
        R = np.atleast_2d(np.unique(source.values.astype(str), axis=0))
        R = pd.DataFrame(R, columns=source.columns, index= pd.Index([":".join(i) for i in R]))
        
        # get the design
        if len(cols) == 0:
            K = pd.DataFrame(index=R.index)
        else:
            K = self.__contrasts__(pd.DataFrame(R[cols]), type=type)
        
        # handle the intercept requirement
        if include_intercept:
            I = pd.DataFrame({'Intercept': np.tile(1, min(1, K.shape[0]))}, index=K.index)
            K = pd.concat([I, K], axis=1)
        
        # return the desing matrices
        return K
    '''
    def __design_matrix__(self, type="sum", include_between=True, include_within=True,
                          include_covariates=True, include_intercept=True):
        """
        obtain the design matrix for every combination allowed by the available variables and
        according to the selected contrasts type.

        Input:
            include_between:    (bool)
                                Should between-factors be included in the matrix?
            
            include_within:     (bool)
                                Should within-factors be included in the matrix?
            
            include_covariates: (bool)
                                Should covariates be included in the matrix?

            include_intercept:  (bool)
                                Should the intercept be included in the matrix?
            
            type:               (str)
                                The type of contrasts to be provided. The options are "sum" or
                                "treat".

        Output:
            K:                  (dict)
                                a dict having each indipendent variable as key which maps a
                                pandas dataframe containing the corresponding design matrix.                               
        """
        
        # check the type
        types = ['sum', 'treat']
        assert np.any([type == i for i in types]), "'type' must be any of " + str(types)

        # check the requirements
        req = {
            "include_between": include_between,
            "include_within": include_within,
            "include_covariates": include_covariates,
            "include_intercept": include_intercept
        }
        for i in req:
            assert req[i] or not req[i], "{} must be a boolean.".format(i)
        
        # get the source factors combinations
        C = []
        IVs = []
        cmb = [i for i in self.__combine__(self.IV)]
        for i in cmb:
            any_between = any_within = any_covariate = False
            cols = i.split(":")
            for j in cols:
                k = pd.DataFrame(self.source[j])
                if not any_between: 
                    any_between = self.__isBetween__(k)
                if not any_within:
                    any_within = not self.__isBetween__(k)
                if not any_covariate:
                    any_covariate = self.__isCovariate__(k)
            if ((include_between or not any_between) and
                (include_within or not any_within) and
                (include_covariates or not any_covariate)):
                C += cols
                IVs += [i]
        C = np.unique(C)
        S = np.atleast_2d(np.unique(self.source[C].values.astype(str), axis=0))
        S = pd.DataFrame(S, columns=C, index=pd.MultiIndex.from_arrays(S.T))

        # get the groups combinations
        K = {}
        for i in IVs:
            cols = i.split(":")
            if np.all([np.any([c == j for c in C]) for j in cols]):
                K[i] = self.__contrasts__(pd.DataFrame(S[cols]), type=type)
        
        # handle the intercept requirement
        if include_intercept:
            ix = K[IVs[0]].index
            K['Intercept'] = pd.DataFrame({'Intercept': np.tile(1, min(1, S.shape[0]))}, index=ix)
        
        # return the desing matrices
        return K
    '''


    def __repr__(self):
        return self.__str__()
    


    def __str__(self):
        def_max_row = pd.get_option("display.max_rows")
        def_precision = pd.get_option("precision")
        pd.set_option("display.max_rows", 999)
        pd.set_option("precision", 3)
        O = "\n\n".join([self.table.__str__(), self.normality_test.__str__()])
        pd.set_option("display.max_rows", def_max_row)
        pd.set_option("precision", def_precision)
        return O