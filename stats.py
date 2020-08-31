


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
from scipy.special import factorial



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



def p_adjust(p):
    """
    return adjusted p-values using different approaches.

    Input:
        p:  (list)
            list of uncorrected p-values

    Output:
        D:  (pandas.DataFrame)
            a dataframe containing the corrected p-values using:
                - Bonferroni
                - Sidak
                - Holm-Bonferroni
                - Holm-Sidak

    References:
        Abdi H., (2010) Encyclopedia of Research Design. Sage. Thousand Oaks, CA.
    """

    # check the entries
    p = np.array([p]).flatten()
    C = len(p)
    I = np.argsort(p)

    # generate the dataframe
    D = pd.DataFrame({'Uncorrected': p}, index=np.arange(C) + 1)
    D.insert(D.shape[1], "Bonferroni", C * p)
    D.insert(D.shape[1], "Sidak", 1 - (1 - p) ** C)
    D.insert(D.shape[1], "Holm-Bonferroni", (C - I) * p)
    D.insert(D.shape[1], "Holm-Sidak", 1 - (1 - p) ** (C - I))

    # return
    return D



def describe(x):
    """
    provide descriptive and distribution statistics for x.

    Input:

        x:  (list)
            a numeric 1D array.

    Output:

        D:  (pandas.DataFrame)
            a row dataframe containing several descriptive statistics about x.
    """

    # get the descriptive stats
    N = len(x)
    avg = np.mean(x)
    std = np.std(x)
    prc = np.quantile(x, [0.25, 0.50, 0.75])
    dmn = np.min(x)
    dmx = np.max(x)
    skw = np.sum((x - avg) ** 3) / ((N - 1) * (std ** 3))
    krt = np.sum((x - avg) ** 4) / (std ** 4)
    krt = krt * N * (N + 1) / (N - 1) / (N - 2) / (N - 3)
    W, p = st.shapiro(x)

    # return the output
    return pd.DataFrame({'N': [N], 'Mean': [avg], 'SD': [std], 'Min': [dmn],
                         '25Q': [prc[0]], '50Q': [prc[1]], '75Q': [prc[2]], 'Max': [dmx],
                         'Skewness': [skw], 'Kurtosis': [krt], 'Shapiro-Wilk': [W], 'P': [p]})



def isCovariate(df):
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



def design(df, vrs, normalize=False):
    """
    matrix locating the unique combinations of factors within a dataframe.

    Input:

        df:         (pandas.DataFrame)
                    the dataframe containing the data

        vrs:        (list)
                    list of column names defining the combinations investigated.

        normalize:  (bool)
                    By default (False) each value that is found is denoted by 1.
                    if normalize is True, the value is 1 / number of occurrences.

    Output:

        O:  (pandas.DataFrame)
            the dataframe containing the df design for the provided variables.
    """

    # check the entries
    assert isinstance(df, pd.DataFrame), "'df' must be a DataFrame."
    assert isinstance(vrs, list), "'vrs' must be a list."
    for v in vrs:
        assert np.any([v == i for i in df.columns]), "{} not found in df.".format(v)
    assert normalize or not normalize, "'normalize' must be a bool object."

    # get the index
    I = df.index

    # get the values
    if len(vrs) > 0:

        # labelizer
        label = lambda x: ":".join(x)

        # get the combinations
        D = pd.DataFrame(df[vrs])
        U = np.unique(D.values.astype(str), axis=0)

        # get the design df
        O = pd.DataFrame(np.zeros((df.shape[0], len(U))), index=I, columns=[label(u) for u in U])
        for u in U:
            n = np.argwhere(D.isin(u).all(1).values.flatten()).flatten()
            O[label(u)].iloc[n] = 1 / (len(n) if normalize else 1)
    else:
        O = pd.DataFrame(index=I)
    return O



def contrasts(df, type="sum"):
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
        if isCovariate(X):
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



def isBetween(df):
    """
    Check whether the current parameter can be considered as a between-subjects variable.

    Input:

        df: (pandas.DataFrame)
            The dataframe representing a variable.

    Output:

        B:  (bool)
            True if the variable can be handled as a betwee-subjects variable. False, otherwise.
    """
    if isCovariate(df):
        return True
    for s in np.unique(df.index.to_numpy()):
        if len(np.unique(df.loc[s].values.astype(str), axis=0)) > 1:
            return False
    return True



def model(source, effect="", include_intercept=True, type="sum"):
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
        K = contrasts(pd.DataFrame(R[cols]), type=type)

    # handle the intercept requirement
    if include_intercept:
        I = pd.DataFrame({'Intercept': np.tile(1, min(1, K.shape[0]))}, index=K.index)
        K = pd.concat([I, K], axis=1)

    # return the desing matrices
    return K



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
        self.SSn = SS_num
        self.MSn = MS_num
        self.SSd = SS_den
        self.MSd = MS_den
        self.eps = eps



class PermutationF(Test):



    def __init__(self, SS_num, DF_num, SS_den, DF_den, pdf, alpha=0.05, two_tailed=True):
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

            pdf:        (list)
                        list of F values on which calculating critical F and p-values.

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

        # check the pdf
        txt = "'pdf' length must be {} or more. {} was provided.".format(1 / alpha, len(pdf))
        assert len(pdf) > 1 / alpha, txt

        # get critical F and p-value
        N = len(pdf)
        if two_tailed:
            a = alpha * 0.5
            f_crit = np.quantile(abs(pdf - np.mean(pdf)), 1 - a)
            p = (np.sum(abs(pdf - np.mean(pdf)) > abs(f - np.mean(pdf))) + 1) / (N + 1)
        else:
            f_crit = np.quantile(pdf, 1 - alpha)
            p = (np.sum(pdf > f) + 1) / (N + 1)

        # create the test
        super(PermutationF, self).__init__(value=f, df=df, crit=f_crit, alpha=alpha,
                                           two_tailed=two_tailed, p=p, name="Permutation F test")

        # add some additional parameters
        self.SSn = SS_num
        self.MSn = MS_num
        self.SSd = SS_den
        self.MSd = MS_den
        self.pdf = pdf



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
        R = effect.residuals(effect.Y)
        SSPE = effect.P.T.dot(R.T.dot(R)).dot(effect.P)
        E = np.real(sl.eigvals(SSPE))
        E = E[E> 0]

        # get the test statistic
        n, p = effect.P.shape
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



class AnovaEffect():



    def __init__(self, src, dv, isB, lbl, X, W, S, P, L, prm):
        """
        basically a container for the data corresponding to each effect of an Anova object.

        Input:

            src:    (pandas.DataFrame)
                    the source data

            dv:     (list)
                    The column corresponding to the dependent variable in source.

            isB:    (bool)
                    is the effect a betwee-subjects effect?

            lbl:    (str)
                    the name of the effect.

            X:      (pandas.DataFrame)
                    the between-subjects regressors.

            W:      (pandas.DataFrame)
                    the within-subjects linear model design.

            S:      (pandas.DataFrame)
                    the subjects linear model design.

            P:      (pandas.DataFrame)
                    the within-subjects design for this effect.

            L:      (pandas.DataFrame)
                    the between-subjects design for this effect.

            prm:    (2D numpy.ndarray)
                    an array where each row contains the combinations of the dependent variable
                    extracted for building the pdf.
        """

        # add the entries
        self.source = src
        self.DV = dv
        self.isBetween = isB
        self.label = lbl
        self.X = X
        self.W = W
        self.S = S
        self.P = P
        self.L = L
        self.permutations = prm

        # inverted covariance
        self.Vi = self.X.T.dot(self.X)

        # covariance
        self.V = pd.DataFrame(sl.inv(self.Vi), index=self.Vi.columns, columns=self.Vi.columns)

        # coefs (raw)
        self.B = self.V.dot(self.X.T)

        # Identity
        I = np.eye(self.source.shape[0])
        self.I = pd.DataFrame(I, index=self.source.index, columns=self.source.index)

        # inverted within-subjects design product
        PPi = sl.inv(self.P.T.dot(self.P))
        self.PPi = pd.DataFrame(PPi, index=self.P.columns, columns=self.P.columns)

        # the DV
        self.Y = pd.DataFrame(self.source[self.DV])



    def SSt(self):
        """
        The total sum of squares
        """
        return np.sum((self.source[self.DV] - self.source[self.DV].mean()).values ** 2)



    def DFt(self):
        """
        The total degrees of freedom.
        """
        return self.source.shape[0] - 1



    def splitWithin(self, Y):
        """
        split Y into its within-subjects components.
        """
        return self.S.T.dot(self.I * Y.values).dot(self.W)



    def coefs(self, Y):
        """
        obtain regression coefficients
        """
        return self.B.dot(self.splitWithin(Y))



    def residuals(self, Y):
        """
        get the residuals of the regression performed providing Y
        """
        return self.splitWithin(Y) - self.X.dot(self.coefs(Y))



    def SSn(self, Y):
        """
        get the Sum of Squares according to the given Y.

        Input:
            Y:      (pandas.DataFrame)
                    the data to be used to calculate the SSn.

        Output:
            SSn:    (float)
                    the sum of squares.
        """
        C = self.L.dot(self.coefs(Y)).dot(self.P)
        SSP = C.T.dot(self.Vi).dot(C)
        return np.sum(np.diag(SSP.dot(self.PPi)))



    def DFn(self):
        """
        get the degrees of freedom of the effect.
        """
        return self.P.shape[1]



    def SSd(self, Y):
        """
        get the Sum of Squares according to the given Y.

        Input:
            Y:      (pandas.DataFrame)
                    the data to be used to calculate the SSn.

        Output:
            SSn:    (float)
                    the sum of squares.
        """
        R = self.residuals(Y)
        SSPE = self.P.T.dot(R.T.dot(R)).dot(self.P)
        return np.sum(np.diag(SSPE.dot(self.PPi)))



    def DFd(self):
        """
        get the degrees of freedom of the error term.
        """
        return np.prod(self.P.shape)



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
        if self.permutations.shape[1] == 0:
            f = F(self.SSn(self.Y), self.DFn(), self.SSd(self.Y), self.DFd(), eps, alpha,
                  two_tailed)

            # index
            cx = [np.tile("F test", 7), ['SSn', 'SSd', 'Statistic', 'DFn', 'DFd', 'Critical', 'P']]
            C = pd.MultiIndex.from_arrays(np.atleast_2d(cx))

            # dataframe
            ln = [f.SSn, f.SSd, f.value, f.df[0], f.df[1], f.crit, f.p]

        # permutation F test
        else:

            # build the pdf
            def permuted_F(p):
                return self.SSn(self.Y.iloc[p]) / self.SSd(self.Y.iloc[p])
            pdf = np.array([permuted_F(p) for p in self.permutations]) * (self.DFd() / self.DFn())

            # get the F test
            f = PermutationF(self.SSn(self.Y), self.DFn(), self.SSd(self.Y), self.DFd(), pdf,
                             alpha, two_tailed)

            # index
            cx = [np.tile("Permutation F test", 8),
                  ['SSn', 'SSd', 'Statistic', 'DFn', 'DFd', 'Perm', 'Critical', 'P']]
            C = pd.MultiIndex.from_arrays(np.atleast_2d(cx))

            # dataframe
            ln = [f.SSn, f.SSd, f.value, f.df[0], f.df[1], len(pdf), f.crit, f.p]

        # return the dataframe
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

        # checks
        not_required = np.any([(self.label == "Intercept"),
                               (self.isBetween or self.DFn() <= 1),
                               (self.permutations.shape[1] > 0)])

        # outcomes
        if not_required:
            ln = np.tile(None, 4)
            jns = None
        else:
            jns = JohnNagaoSugiura(self, alpha, two_tailed)
            ln = [jns.value, jns.df, jns.crit, jns.p]

        # dataframe
        test = pd.DataFrame(ln, index=C, columns=[self.label]).T


        #* GREENHOUSE-GEISSER SPHERICITY CORRECTION

        # index
        cx = [np.tile("Greenhouse-Geisser", 5), ['Epsilon', 'DFn', "DFd", "Crit", 'P']]
        C = pd.MultiIndex.from_arrays(np.atleast_2d(cx))

        # outcomes
        if not_required:
            ln = np.tile(None, 5)
        else:
            gg_eps = self.epsilon_GG()
            gg_f = F(self.SSn, self.DFn, self.SSd, self.DFd, gg_eps, alpha, two_tailed)
            ln = [gg_eps, gg_f.df[0],  gg_f.df[1], gg_f.crit, gg_f.p]

        # dataframe
        gg = pd.DataFrame(ln, index=C, columns=[self.label]).T


        #* HUYNH-FELDT SPHERICITY CORRECTION

        # index
        cx = [np.tile("Huynh-Feldt", 5), ['Epsilon', 'DFn', "DFd", "Crit", 'P']]
        C = pd.MultiIndex.from_arrays(np.atleast_2d(cx))

        # outcomes
        if not_required:
            ln = np.tile(None, 5)
        else:
            hf_eps = self.epsilon_HF()
            hf_f = F(self.SSn, self.DFn, self.SSd, self.DFd, hf_eps, alpha, two_tailed)
            ln = [hf_eps, hf_f.df[0], hf_f.df[1], hf_f.crit, hf_f.p]

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
        cx = [np.tile('Brown-Forsythe', 5), ["Statistic", 'DFn', "DFd", "Crit", 'P']]
        C = pd.MultiIndex.from_arrays(cx)

        # values
        if not self.isBetween or self.permutations.shape[1] > 0:
            ln = np.tile(None, 5)
        else:
            BF = BrownForsythe(self.source, self.DV, self.label.split(":"), alpha, two_tailed)
            ln = [BF.value, BF.DF_num, BF.DF_den, BF.crit, BF.p]

        # return the dataframe
        return pd.DataFrame(ln, index=C, columns=[self.label]).T



    def epsilon_GG(self):
        """
        return the Greenhouse-Geisser epsilon for sphericity correction.
        """
        if self.isBetween:
            return 1
        else:

            # get the positive eigenvalues from SSPE
            R = self.residuals(self.Y)
            SSPE = self.P.T.dot(R.T.dot(R)).dot(self.P)
            E = np.real(sl.eigvals(SSPE))
            E = E[E > 0]

            # return the Greenhouse-Geisser epsilon
            p = self.DFn()
            return ((np.sum(E) / p) ** 2) / (np.sum(E ** 2) / p)



    def epsilon_HF(self):
        """
        return the Huynh-Feldt epsilon for sphericity correction.
        """
        if self.isBetween:
            return 1
        else:

            # get the positive eigenvalues from SSPE
            R = self.residuals(self.Y)
            SSPE = self.P.T.dot(R.T.dot(R)).dot(self.P)
            E = np.real(sl.eigvals(SSPE))
            E = E[E > 0]

            # return the Huynh-Feldt epsilon
            n, p = self.P.shape
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
        ln = self.SSn(self.Y) / np.array([self.SSt(), self.SSn(self.Y) + self.SSd(self.Y),
                                          self.SSn(self.Y) + SS_error])
        return pd.DataFrame(ln, columns=[self.label], index=C).T



class Anova(LinearRegression):



    def __init__(self, source, dv, iv, subjects=None, alpha=0.05, two_tailed=True, n_perm=0, exclude=[]):
        """
        Generate an Anova class object.

        Input:
            source:         (pandas.DataFrame)
                            the dataframe where all variables are stored.

            dv:             (str)
                            the name of the column defining the dependent variable of the model.

            iv:             (list)
                            the list of column names containing the indipendent variables.

            subjects:       (str or None)
                            If none, the factors are treated as "between-only" factors. Conversely,
                            if a list or a 1D ndarray is provided, the values are used to define
                            within-subjects errors.
                            If such value is provided, it is also added to source.

            alpha:          (float)
                            the level of significance.

            two_tailed:     (bool)
                            should the p-value be calculated from a two-tailed distribution?

            n_perm:         (int)
                            the number of permutations to be used for drawing the test Probability
                            Density Function (PDF).
                            If 0 (Default) p values of the F tests are calculated using the F
                            distribution with degrees of freedom calculated from data.
                            If -1, all possible permutations of the the available data are used to
                            obtain the PDF which, in turn, is used to calculated the p values.
                            Please note that this setting might result in an EXTREMELY long
                            computational time.
                            If a positive int is provided, the PDF is obtained throught the given
                            number of random permutations.

            exclude:        (list)
                            a list of str objects defining the effects which should not be part of
                            the model. This is useful the user aims at excluding specific main or
                            interaction effects from the model. Please note that the interaction
                            effects must be provided as the names of the effects involved separated
                            by a ":". E.g. --> Main effects: "A", "B" --> Interaction: "A:B"
        """

        #* DATA PREPARATION

        # check the variables
        assert isinstance(source, pd.DataFrame), "source must be a pandas.DataFrame instance."
        assert isinstance(iv, list), "'iv' must be a list instance."
        assert isinstance(dv, list), "'dv' must be a list instance."
        source_missing = "{} not found in 'source'."
        for i in iv + dv:
            assert np.any([i == j for j in source]), source_missing.format(i)
        self.DV = dv
        self.IV = iv

        # check alpha
        assert isinstance(alpha, float), "alpha must be a float in the (0, 1) range."
        assert alpha > 0 and alpha < 1, "alpha must be a float in the (0, 1) range."
        self.alpha = alpha

        # check two_tailed
        assert isinstance(two_tailed, bool), "'two_tailed' must be a bool object."
        self.two_tailed = two_tailed

        # check the subjects
        if subjects is not None:
            assert np.any([subjects == i for i in source]), source_missing.format(subjects)
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

        # check n_perm
        assert isinstance(n_perm, int), "'n_perm' must be an int object."
        self.max_perm = factorial(self.source.shape[0])
        if n_perm > self.max_perm or n_perm < 0:
            self.n_perm = self.max_perm
        else:
            self.n_perm = n_perm

        # draw the random permutation index
        if self.n_perm > 0:
            self.permutations = np.atleast_2d(np.arange(self.source.shape[0]))
            np.random.seed()
            while len(self.permutations) <= self.n_perm:
                new = np.atleast_2d(np.random.permutation(self.source.shape[0]))
                self.permutations = np.vstack([self.permutations, new])
                self.permutations = np.unique(self.permutations, axis=0)
            self.permutations = self.permutations[1:]
        else:
            self.permutations = np.atleast_2d([])

        # separate between and within factors
        BV = []
        WV = []
        for i in self.source:
            if not np.any([i == j for j in dv]):
                if isBetween(pd.DataFrame(self.source[i])):
                    BV += [i]
                else:
                    WV += [i]
        self.between = BV
        self.within = WV
        self.covariates = [i for i in self.IV if isCovariate(pd.DataFrame(self.source[i]))]

        # check the exclusions
        assert isinstance(exclude, list), "'exclude' must be a list object."
        for ex in exclude:
            ex_ck = [np.any([i == j for j in self.source.columns]) for i in ex.split(":")]
            assert np.all(ex_ck), "{} or one of its components was not found in source.".format(ex)
        self.exclude = exclude

        # initialize the object
        X = self.__combine__(self.between + self.within)
        X = pd.concat([contrasts(self.source[i.split(":")]) for i in X], axis=1)
        Y = pd.DataFrame(self.source[dv])
        super(Anova, self).__init__(Y, X)


        #* DESIGN PARAMETERS

        # get the between and within subjects variable combinations
        BV_combs = {'Intercept': ["Intercept"], **self.__combine__(self.between)}
        WV_combs = {'Intercept': ["Intercept"], **self.__combine__(self.within)}

        # create the between-subjects data
        GRP = self.source.copy()
        GRP.insert(0, "SBJ", self.source.index.to_numpy())

        # between
        if len(self.between) > 0:
            X = design(GRP, self.between)
        else:
            X = pd.DataFrame(index=GRP.index)
        X.insert(0, "Intercept", np.tile(1, GRP.shape[0]))
        X = contrasts(design(GRP, ["SBJ"], normalize=True).T.dot(X))

        # within
        W = design(GRP, self.within)

        # subjects
        S = design(GRP, ["SBJ"])


        #* EFFECTS CALCULATION

        self.effects = {}
        I = pd.DataFrame(np.eye(X.shape[1]), index=X.columns, columns=X.columns)
        for j, b in enumerate(BV_combs):

            # between-subjects linear function
            L = pd.DataFrame(I[np.concatenate([k.split(":") for k in BV_combs[b]]).flatten()]).T

            # design matrix of the within effects
            for i, w in enumerate(WV_combs):
                D = pd.DataFrame(self.source[self.within])
                if w == "Intercept":
                    P = model(D, include_intercept=True)
                else:
                    P = model(D, w, include_intercept=False)

                # between
                isB = (w == "Intercept") & (b != "Intercept")

                # label
                lbl = b if w == "Intercept" else (w if b == "Intercept" else ":".join([b, w]))

                # store the effect
                self.effects[lbl] = AnovaEffect(GRP, self.DV, isB, lbl, X, W, S, P, L,
                                                self.permutations)



    def anova_table(self, digits=4):
        """
        Return a pandas.DataFrame containing the Anova analysis of the entered model.

        Input:

            digits: (int)
                    the number of decimals to be used in representing the outcomes.

        Output:

            D:      (pandas.DataFrame)
                    a dataframe containing all the outcomes.
        """

        # get the total error SS to calculate the generalized effect sizes
        SSe = np.sum([self.effects[e].SSd(self.effects[e].Y) for e in self.effects])

        # create the table
        table = pd.DataFrame()
        for e in self.effects:
            contents = [self.effects[e].variance_test(), self.effects[e].F_test(),
                        self.effects[e].sphericity_test(), self.effects[e].effect_sizes(SSe)]
            table = table.append(pd.concat(contents, axis=1))

        # drop empty columns
        table = table.dropna(axis=1, how='all')

        # return the outcomes rounded to the desired decimal
        return table.apply(self.__rnd__, decimals=digits)



    def descriptive_table(self, digits=4):
        """
        Return a table containing several descriptive statistics calculated from both the
        residuals and all the model combinations.
        If any covariate exists in the model, fixed values are used to obtain descriptive stats.

        Input:

            digits: (int)
                    the number of decimals to be used in representing the outcomes.

        Output:

            D:      (pandas.DataFrame)
                    a dataframe containing all the outcomes.
        """

        # get descriptive and marginal means stats
        M = pd.DataFrame()
        D = pd.DataFrame()
        for e in [i for i in self.effects if i != "Intercept"]:

            # get the emmeans but keep only the columns of insterest
            emm_cols = ["Estimate", "Standard error",
                        "{:0.0f}% C.I. (inf)".format(100 * (1 - self.alpha)),
                        "{:0.0f}% C.I. (sup)".format(100 * (1 - self.alpha))]
            em = self.__emmeans__(self.effects[e])[emm_cols]
            ex = np.atleast_2d([[e, i] for i in em.index])
            em.index = pd.MultiIndex.from_arrays(ex.T)

            # store the marginal means
            M = M.append(em)

            # get the unique combinations for each effect
            combs = np.atleast_2d(np.unique(self.source[e.split(":")].values.astype(str), axis=0))

            # get the data corresponding to each combination
            for c in combs:

                # data
                dv = self.source.loc[self.source[e.split(":")].isin(c).all(1)][self.DV].values.flatten()

                # get the descriptive stats
                K = describe(dv)
                K.index = pd.MultiIndex.from_arrays(np.atleast_2d([[e, ":".join(c)]]).T)
                D = D.append(K)

        # get descriptive statistics of the residuals
        R = describe(self.residuals().values.flatten())
        R.index = pd.MultiIndex.from_arrays(np.atleast_2d([["Residuals", ""]]).T)
        D = D.append(R)

        # adjust the column index
        C = [["Descriptive stats", i] for i in D.columns]
        D.columns = pd.MultiIndex.from_arrays(np.atleast_2d(C).T)
        F = [["Estimated Marginal Means", i] for i in M.columns]
        M.columns = pd.MultiIndex.from_arrays(np.atleast_2d(F).T)

        # merge and return
        G = pd.concat([D, M], axis=1)
        return G.apply(self.__rnd__, decimals=digits)



    def contrasts_table(self, digits=4):
        """
        Return pairwise contrasts based on estimated marginal means for each effect.

        Input:

            digits: (int)
                    the number of decimals to be used in representing the outcomes.

        Output:

            D:      (pandas.DataFrame)
                    a dataframe containing all the outcomes.
        """

        # iterate each effect
        EM = pd.DataFrame()
        for e in [i for i in self.effects if i != "Intercept"]:

            # get the linear function and the scaled covariance matrix for the effect
            L = self.__linfun__(self.effects[e])

            # obtain the contrast matrix
            J = np.atleast_2d(np.unique(self.source[e.split(":")].values.astype(str), axis=0))
            M = pd.DataFrame(index=L.index)
            for i, j in enumerate(J[:-1]):
                for z in J[(i + 1):]:

                    # check if the actual combination is a required contrast
                    if np.sum([1 if k != f else 0 for k, f in zip(j, z)]) == 1:

                        # create the contrast
                        col = " - ".join([":".join(j), ":".join(z)])
                        cmj = pd.DataFrame(np.zeros((L.shape[0], 1)), index=L.index, columns=[col])
                        cmj.loc[":".join(j)] = 1
                        cmj.loc[":".join(z)] = -1

                        # add the contrast
                        M = pd.concat([M, cmj], axis=1)

            # adjust the linear function for the contrasts
            C = M.T.dot(L)

            # get the emmeans but keep only the columns of insterest
            emm_cols = ["Estimate", "Standard error",
                        "{:0.0f}% C.I. (inf)".format(100 * (1 - self.alpha)),
                        "{:0.0f}% C.I. (sup)".format(100 * (1 - self.alpha)),
                        "T stat", "DF", "T crit", "P", "P adj. (Holm-Sidak)"]
            em = self.__emmeans__(self.effects[e], C)[emm_cols]
            ix = np.atleast_2d([[e, i] for i in em.index])
            em.index = pd.MultiIndex.from_arrays(ix.T)

            # store the output
            EM = EM.append(em)

        # return the outcomes rounded to the desired decimal
        return EM.apply(self.__rnd__, decimals=digits)



    def __rnd__(self, x, decimals=4):
        """
        internal function used to round values.

        Input:

            x:          (object)
                        an object to be rounded

            decimals:   (int)
                        the number of decimals.

        Output:

            r:  (object)
                the object rounded (where possible) or the same object entered, otherwise.
        """
        try:
            return np.around(x, decimals)
        except Exception:
            return x



    def __linfun__(self, E):
        """
        get the linear function corresponding to the given effect.

        Input:
            E:  (AnovaEffect)
                The effect about which the emmeans have to be calculated.

        Output:
            M:  (pandas.DataFrame)
                a dataframe containing the linear function for the effect.
        """

        # obtain the linear function
        I = np.unique(self.source[E.label.split(":")].values.astype(str), axis=0)
        I = pd.Index([":".join(i) for i in I])
        C = self.coefs.index
        L = pd.DataFrame(np.zeros((len(I), len(C))), index=I, columns=C)
        P = model(self.source[E.label.split(":")], E.label, include_intercept=True)
        L.loc[P.index, P.columns] = P.values
        return L



    def __emmeans__(self, E, L=None):
        """
        return the estimated marginal means for an effect given through its linear function and its
        scaled covariance matrix.

        Input:

            L:  (pandas.DataFrame)
                the dataframe containing the linear function of an effect. Typically it is the output
                of the __linfun__ function.

            V:  (pandas.DataFrame)
                the dataframe containing the covariance matrix of the coefficients scaled by the
                error of an effect. Typically it is the output of the __cov_scaled__ function.

        Output:
            M:  (pandas.DataFrame)
                a dataframe containing the emmeans for the effect.
        """

        def cov_scaled(Y):
            """
            return the covariance matrix scaled by the mean error of the effect.

            Input:
                Y:  (pandas.DataFrame)
                    the dependent variable on which calculating the sum of squares.

            Output:
                M:  (pandas.DataFrame)
                    a dataframe containing the covariance matrix scaled on the effect.
            """
            if E.label != "Intercept":
                C = E.P.columns
                S = Vi.copy()
                S.loc[C, C] *= (E.SSd(Y) / E.DFd())
            return S


        def get_MM(Y):
            """
            obtain estimated marginal means according to the provided dependent variable and the current model.

            Input:

                Y:  (pandas.DataFrame)
                    the dataframe containing the dependent variable data.

            Output:
                MM: (pandas.DataFrame)
                    the dataframe containing the estimated standard errors.
            """
            em = K.dot(Y.values).T
            em.index = pd.Index(['Estimate'])
            return em


        def get_SE(Y):
            """
            obtain standard errors according to the provided dependent variable and the current model.

            Input:

                Y:  (pandas.DataFrame)
                    the dataframe containing the dependent variable data.

            Output:
                SE: (pandas.DataFrame)
                    the dataframe containing the estimated standard errors.
            """
            # obtain the standard errors
            M = L.dot(cov_scaled(Y)) * L
            se = pd.DataFrame(M.sum(1)).T.apply(np.sqrt)
            se.index = pd.Index(['Standard error'])
            return se


        def get_T(Y):
            """
            obtain T values according to the provided dependent variable and the current model.

            Input:

                Y:  (pandas.DataFrame)
                    the dataframe containing the dependent variable data.
                
            Output:
                T: (pandas.DataFrame)
                    the dataframe containing the T values.
            """
            T = get_MM(Y) / get_SE(Y).values
            T.index = pd.Index(['T stat'])


        # get the linear function
        if L is None:
            L = self.__linfun__(E)

        # get the DV
        Y = pd.DataFrame(self.source[self.DV])

        # get the unscaled coefficients
        V = self.cov_unscaled()
        XX = self.X.copy()
        XX.insert(0, "Intercept", np.tile(1, XX.shape[0]))
        K = L.dot(V).dot(XX.T)
        Vi = V.copy()
        Vi.loc["Intercept", "Intercept"] *= self.effects["Intercept"].SSd(Y) / self.effects["Intercept"].DFd()

        # get the estimates
        em = get_MM(Y)

        # get the standard errors
        se = get_SE(Y)

        # T value
        tv = get_T(Y)

        # build the pdf
        if np.all(abs(L.values) <= 1):
            A = (design(self.source[E.label.split(":")], E.label.split(":")) * em.values).sum(1).values
        else:
            A = np.zeros(Y.shape)
        pdf = pd.concat([get_T(Y.iloc[p] - A.values) for p in self.permutations], axis=0)
        if self.two_tailed and pdf.shape[0] > 0:
            pdf = pdf.abs()

        # get T-critial
        tc = em.copy()
        al = 1 - (self.alpha * (0.5 if self.two_tailed else 1))
        if pdf.shape[0] == 0:

            # get the degrees of freedom used to calculate confidence intervals
            dfN = pd.DataFrame(np.zeros((L.shape[0], 1)), index=L.index, columns=["DFc"])
            dfD = pd.DataFrame(np.zeros((L.shape[0], 1)), index=L.index, columns=["DFc"])
            M = L.dot(cov_scaled(Y)) * L
            for e in self.effects:
                ii = self.effects[e].P.columns
                dfe = self.effects[e].DFd()
                dfN += pd.DataFrame(pd.DataFrame(M[ii]).sum(1), columns=["DFc"])
                dfD += pd.DataFrame(pd.DataFrame(M[ii]).sum(1) ** 2 / dfe, columns=["DFc"])
            dfc = (dfN ** 2 / dfD).T

            # critical T value
            for t in tc:
                tc.loc[tc.index, t] = st.t.isf(al, dfc.loc[tc.index, t])

        else:

            # critical T
            for t in tc:
                tc.loc[tc.index, t] = np.quantile(pdf[t].values.flatten(), al)
        tc.index = pd.Index(['T crit'])

        # confidence intervals
        ci_inf = em.copy()
        ci_sup = em.copy()
        for t in ci_inf:
            c = tc.loc[tc.index, t].values * se.loc[se.index, t].values
            ci_inf.loc[ci_inf.index, t] = em.loc[em.index, t].values - c
            ci_sup.loc[ci_sup.index, t] = em.loc[em.index, t].values + c
        ci_inf.index = pd.Index(["{:0.0f}% C.I. (inf)".format(100 * (1 - self.alpha))])
        ci_sup.index = pd.Index(["{:0.0f}% C.I. (sup)".format(100 * (1 - self.alpha))])

        # get the effect degrees of freedom
        df = em.copy()
        df.loc[df.index] = E.DFd()
        df.index = pd.Index(["DF"])

        # get the test p values
        pv = tv.copy()
        for t in tv:
            v = abs(tv.loc[tv.index, t]) if self.two_tailed else tv.loc[tv.index, t]
            if pdf.shape[0] == 0:
                pv.loc[pv.index, t] = st.t.sf(v, df.loc[df.index, t])
            else:
                pv.loc[pv.index, t] = (np.sum(pdf[t].values > v.values) + 1) / (pdf.shape[0] + 1)
        pv.index = pd.Index(['P'])

        # get the corrected p-values
        pa = p_adjust(pv.values.flatten())["Holm-Sidak"].values
        pa = pd.DataFrame(pa, index=L.index, columns=["P adj. (Holm-Sidak)"]).T

        # return the table
        return pd.concat([em, se, ci_inf, ci_sup, tv, df, tc, pv, pa], axis=0).T



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
                contrasts(self.source[list(j)]).columns.to_numpy().tolist()
                for i in np.arange(len(x)) + 1 for j in it.combinations(x, i)
                if not np.any([":".join(list(j)) == k for k in self.exclude])}



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