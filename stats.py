


########    IMPORTS    ########



import math
import numpy as np
import pandas as pd
import joblib as jl
import itertools as it
import scipy.stats as ss
import warnings
import utils as pu



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


    def __init__(self, value, name):
        """
        Generic class method which contains the results of Effect Size measurement.

        Input:
            value:  (float)
                    the effect size value.

            name:   (str)
                    the name of the effect size measure.
        """

        self.name = name
        self.value = float(value)

   

    def to_df(self):
        """
        Create a column DataFrame containing all the outcomes of the EffectSize object.
        """
        return pd.DataFrame([getattr(self, i) for i in self.attributes()], index = self.attributes(), columns=[0])



    def copy(self):
        """
        create a copy of the current Test object.
        """
        return Test(**{i: getattr(self, i) for i in self.attributes()})



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



    def __init__(self, value, p, name, alpha, crit, **kwargs):
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
        """

        super(Test, self).__init__(value, name)
        self.crit = float(crit)
        self.alpha = float(alpha)
        self.p = float(p)
        
        # add all the other parameters
        for i in kwargs:
            setattr(self, i, kwargs[i])



########    IMPLEMENTATION CLASSES    ########



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
        v = ss.f.sf(f, *df_f)

        # get the critical T2 test
        f_crit = ss.f.isf(alpha, *df_f)
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
        return float(a.dot(np.linalg.inv(K)).dot(a.T) * n)



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
            return float((na * nb) / (na + nb) * (ab - aa).dot(np.linalg.inv(Kp)).dot((ab - aa).T))
        except Exception:
            return float((na * nb) / (na + nb) * (ab - aa).dot(np.linalg.pinv(Kp, hermitian=True)).dot((ab - aa).T))



class T(Test):



    def __init__(self, source, data, groups=None, mu=0, paired=True, alpha=0.05, two_tailed=True, verbose=True):
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
                        If both A and B are provided, paired=True will return a paired T test. Otherwise,
                        a two-sample test is performed. If B is None, paired is ignored.

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
                if BrownForsythe(source=source, data=data, groups=groups, alpha=alpha).p < alpha:
                    if verbose:
                        warnings.warn("groups have unequal variance. Welch's T test is used.")
                    name = "Welch's T"
                    df = ((sa / na + sb / nb) ** 2) / ((((sa / na) ** 2) / (na - 1)) + (((sb / nb) ** 2) / (nb - 1)))
                    Sp = np.sqrt(sa / na + sb / nb)
                
                # TWO SAMPLES T
                else:
                    name = "T two samples"
                    df = na + nb - 2
                    Sp = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2)) * np.sqrt(1 / na + 1 / nb)
                t = (np.mean(df_data[0]) - np.mean(df_data[1])) / Sp
                
        # get the critical T test
        a = alpha * (0.5 if two_tailed else 1)
        t_crit = ss.t.isf(a, df)
        
        # adjust the test value in case of a two-tailed test
        if two_tailed:
            t = abs(t)

        # get the p value
        v = ss.t.sf(t, df)
        v = np.min([1, 2 * v]) if two_tailed else v

        # create the test
        super(T, self).__init__(value=t, df=df, crit=t_crit, alpha=alpha, two_tailed=two_tailed, p=v, name=name)



class BrownForsythe(Test):



    def __init__(self, source, data, groups, alpha=0.05):
        """
        tests the null hypothesis that the population variances are equal.

        Input:
            source:     (pandas.DataFrame)
                        the dataframe containing the data

            data:       (str, 1D array or list)
                        The names of the source columns containing the data to be investigated.

            groups:     (str)
                        The name of the source column containing the grouping variable.

            alpha:      (float)
                        the level of significance.

        Output:
            A Test instance object containing the outcomes of the test.
        """

        # get the data
        df_data = split_data(source, data, groups)[0]

        # split the data in groups
        samples = []
        sizes = []
        means = []
        for i in df_data:
            samples += [abs(i - np.median(i))]
            sizes += [i.shape[0]]
            means += [np.mean(samples[-1])]
        
        # get the data concerning the whole dataset
        N = np.sum(sizes)                                                 # data size
        Z = np.sum([i * v for i, v in zip(means, sizes)]) / N             # grand mean  
        P = len(df_data)                                                  # number of groups

        # get the F statistic
        F = (N - P) / (P - 1) * np.sum([i * ((v - Z) ** 2) for i, v in zip(sizes, means)])
        F /= np.sum([np.sum((i - v) ** 2) for i, v in zip(samples, means)])
        df = (P - 1, N - P)

        # get the p-value
        p = ss.f.sf(F, *df)

        # get the critical F value
        F_crit = ss.f.isf(alpha, *df)

        # store the outcomes
        super(BrownForsythe, self).__init__(value=F, df=df, crit=F_crit, alpha=alpha, p=p, name="Brown-Forsythe")



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