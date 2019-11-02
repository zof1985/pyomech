# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:12:24 2018

@author: lzoffoli
"""



def cohen_d(A, B):
    """
    Calculate the Cohen's d effect size between A and B

    Input:
        A: (ndarray)
            a numpy array reflecting the data of one group. The rows are considered to be the samples while the colums
            are the nodes.

        B: (ndarray)
            a numpy array reflecting the data of the other group.  The rows are considered to be the samples while 
            the colums are the nodes.

        axis: (int)
            the axis of A and B along which perform the calculation

    Output:
        D: (ndarray)
            a ndarray with the effect sizes.
    """

    # dependancies
    import numpy as np
    from pybiomech.utils import classcheck

    # check the entered data
    classcheck(A, ['ndarray'])
    A = np.atleast_2d(A).T if A.ndim == 1 else A
    classcheck(B, ['ndarray'])
    B = np.atleast_2d(B).T if B.ndim == 1 else B

    # get the data
    nA, cA = A.shape
    nB, cB = B.shape

    # get the pooled SD
    pSD = np.sqrt(((nA - 1) * np.var(A, 0) + (nB - 1) * np.var(B, 0)) / (nA + nB - 2))

    # return the effect size
    return (np.mean(A, 0) - np.mean(B, 0)) / pSD
    
    
def __map_significance__(snpm_obj):
    """
    map the nodes p-values in a non-parametric SPM1D object resulting from spm1d_anova or spm1d_t
    Input:
        snpm_obj: (SnPM object or dict)
            the result of st.spm1d_anova or st.spm1d_t methods.

    Output:
        Generate a 1D array with length equal to the nodes of the object. For each node, the array returns zero if it
        was found non-significantfly different, and 1 otherwise.
    """

    # dependancies
    import numpy as np

    # get the map
    control = {}
    for eff in snpm_obj:
        control[eff] = np.ones_like(snpm_obj[eff].z)
        try:
            for cl in snpm_obj[eff].clusters:
                try:
                    p = cl.P
                except Exception:
                    p = cl.p
                control[eff][np.arange(int(np.floor(cl.endpoints[0])), int(np.ceil(cl.endpoints[1])) + 1)] = p
        except Exception:
            try:
                p = snpm_obj[eff].P
            except Exception:
                p = snpm_obj[eff].p
            control[eff] = np.array([p]).flatten()

    # return the map
    return control


def spm1d_anova(data, within=None, between=None, alpha=0.05, perm=None, test_accuracy=True, accuracy_trials=100,
                aov_kwargs={}, inf_kwargs={}):
    '''
    a wrapper of the anova-family functions of the spm1d package

    Input:
        data: (pandas.DataFrame)
            a dataframe containing all the data (nodes).
        between: (None, pandas.DataFrame)
            a dataframe with the between factors.
        within: (None, pandas.DataFrame)
            a DataFrame with the within-subjects factors.
        alpha: (float)
            the level of significance
        perm: (int, None)
            the number of permutations to be used for the non-parametric anova. If None, parametric analysis is
            performed, otherwhise non-parametric analysis with n=perm permutations is computed. If perm=-1, all the
            possible permutations are used.
        test_accuracy: (Boolean)
            the accuracy of the statistics is evaluated (it requires high computational cost). Accuracy quantifies the
            average variability in the determination of the clusters length and p-values. To this purpose, the test
            statistics is repeated several times and the average variability is calculated.
        accuracy_trials: (int)
            the number of test replications to be used as accuracy measure.
        aov_kwargs: (any)
            a dict of arguments passed to the spm1d anova functions.
        inf_kwargs: (any)
            a dict of arguments passed to the inference of the anova test.

    Output:
        anova: (pandas.DataFrame)
            a pandas dataframe with the output of the test.
    '''

    # get dependancies
    import numpy as np
    import pandas as pd
    import spm1d.stats as st
    from itertools import combinations

    # check the entered data
    assert data.__class__.__name__ == "DataFrame", "'data' must be a pandas.DataFrame object."
    assert within.__class__.__name__ in ['NoneType', 'DataFrame'], "'within' must be a pandas.DataFrame or None."
    assert between.__class__.__name__ in ['NoneType', 'DataFrame'], "'between' must be a pandas.DataFrame or None."
    assert alpha.__class__.__name__[:5] == 'float', "'alpha' must be a float object."
    assert 0 < alpha < 1, "'alpha' must lie in the (0, 1) range."
    assert perm is None or perm.__class__.__name__[:3] == 'int', "'perm' must be None or an int."
    assert test_accuracy or not test_accuracy, "'test_accuracy' must be a bool object."
    assert accuracy_trials.__class__.__name__[:3] == "int", "'accuracy_trials' must be an int object."

    # check the suitability of the factors to the spm1d anova
    if within is None: within = pd.DataFrame(index=data.index.to_numpy())
    if between is None: between = pd.DataFrame(index=data.index.to_numpy())
    assert 0 < between.shape[1] + within.shape[1] < 4, 'spm1d anova currently supports only a 3 or less factors.'
    assert between.shape[0] == data.shape[0], "'data', and 'between' must have equal row number."
    assert within.shape[0] == data.shape[0], "'data', and 'within' must have equal row number."

    # generate a storing dict for all the parameters to be passed to the spm1d anova method
    dd = {'Y' if perm is None else 'y': np.squeeze(data.values)}

    # create the between subjects surrogate ndarray
    fctrs = ['A', 'B', 'C'][:(len(between) + len(within))]
    b_f = between.copy()
    for f in b_f:
        for i, v in enumerate(np.unique(b_f[f].values)):
            b_f.loc[b_f[f].values.flatten() == v, f] = i
        dd[fctrs[len(dd) - 1]] = b_f[f].values.flatten()

    # create the within subjects surrogate ndarray
    w_f = within.copy()
    for f in w_f:
        for i, v in enumerate(np.unique(w_f[f].values)):
            w_f.loc[w_f[f].values.flatten() == v, f] = i
        dd[fctrs[len(dd) - 1]] = w_f[f].values.flatten()

    # get the subjects
    if within.shape[1] > 0:
        idx = data.index.to_numpy()
        assert np.all([i in idx for i in between.index.to_numpy()]), "'between' must have the same index than 'data'."
        assert np.all([i in idx for i in within.index.to_numpy()]), "'within' must have the same index than 'data'."
        dd['SUBJ'] = np.array([np.argwhere(i == np.unique(idx)).flatten() for i in idx]).flatten()

    # add the extra arguments to be passed to the spm1d anova method
    dd.update(aov_kwargs)

    # generate the dict with the options for the test inference
    ii = {'alpha': alpha}
    ii.update({'iterations': perm, 'force_iterations': True} if perm is not None else {'interp': False})
    ii.update(inf_kwargs)

    # get the correct ANOVA method
    fun = st if perm is None else st.nonparam
    if between.shape[1] == 1 and within.shape[1] == 0: fun = fun.anova1
    elif between.shape[1] == 0 and within.shape[1] == 1: fun = fun.anova1rm
    elif between.shape[1] == 2 and within.shape[1] == 0: fun = fun.anova2
    elif between.shape[1] == 1 and within.shape[1] == 1: fun = fun.anova2onerm
    elif between.shape[1] == 0 and within.shape[1] == 2: fun = fun.anova2rm
    elif between.shape[1] == 3 and within.shape[1] == 0: fun = fun.anova3
    elif between.shape[1] == 2 and within.shape[1] == 1: fun = fun.anova3onerm
    elif between.shape[1] == 1 and within.shape[1] == 2: fun = fun.anova3tworm
    elif between.shape[1] == 0 and within.shape[1] == 3: fun = fun.anova3rm

    # generate the accuracy pool
    pool = 1 + (accuracy_trials  if test_accuracy else 0)

    # perform the test
    tests = []
    for i in np.arange(pool):
        test = fun(**dd).inference(**ii)
        if not test.__class__.__name__ in ["SnPMFiList", "SPMFiList", "SnPMFiList0D", "SPMFiList0D"]: test = [test]

        # map the effects
        effects = np.append([i for i in between.columns], [j for j in within])
        effects = [" x ".join(j) for i in np.arange(len(fctrs)) for j in combinations(effects, i + 1)]
        effects = {v: test[i] for i, v in enumerate(effects)}

        # store the outcome of the test
        if i == 0:
            out = effects.copy()
        tests += [__map_significance__(effects)]

    # generate the accuracy table (for each effect)
    acc = {v: (1 - np.mean([abs(tests[0][v] - i[v]) for i in tests[1:]])) if test_accuracy else None for v in effects}

    # generate the output dataframe
    df = pd.DataFrame()
    nodes = [i for i in data.columns]
    for i, v in enumerate(out):
        eff_df = __spm1d_df__(out[v], nodes).drop("index", axis=1, errors="ignore")
        eff_df.insert(0, 'Accuracy', np.tile(acc[v], eff_df.shape[0]))
        eff_df.insert(0, 'Effect', np.tile(v, eff_df.shape[0]))
        df = df.append(eff_df)
    df.index = np.arange(df.shape[0])
    return effects, df



def spm1d_t(data, factor, alpha=0.05, perm=None, roi=None, test_accuracy=True, accuracy_trials=100, inf_kwargs={}):
    """
    a wrapper of the t-family functions of the spm1d package

    Input:
        data: (pandas.DataFrame)
            a dataframe containing all the data
        factor: (pandas.DataFrame, pandas.Series)
            a dataframe with 1 column containing the factors to be investigated. According to factor 3 distinct
            options of tests can be provided:
                paired t test:
                    2 factors with each value in index repeated on each factor
                t test:
                    2 factors with distinct index value between the factor
                one sample t test:
                    1 factor (all elements in factors are equal). In this case the mean value is used as reference to
                    compute a one-sample t test.
        alpha: (float)
            the level of significance
        perm: (int, None)
            the number of permutations to be used for the non-parametric anova. If None, parametric analysis is
            performed, otherwhise non-parametric analysis with n=perm permutations is computed. If perm=-1, all the
            possible permutations are used.
        roi: (list, ndarray, None)
            None or a list of nodes where the analysis has to be restricted.
        test_accuracy: (Boolean)
            the accuracy of the statistics is evaluated (it requires high computational cost). Accuracy quantifies the
            average variability in the determination of the clusters length and p-values. To this purpose, the test
            statistics is repeated several times and the average variability is calculated.
        accuracy_trials: (int)
            the number of test replications to be used as accuracy measure.
        inf_kwargs: (any)
            a dict of arguments passed to the inference of the anova test.

    Output:
        anova: (pandas.DataFrame)
            a pandas dataframe with the output of the test.
    """

    # get dependancies
    import numpy as np
    import spm1d.stats as st

    # check the entered data
    assert data.__class__.__name__ == "DataFrame", "'data' must be a pandas.DataFrame object."
    assert alpha.__class__.__name__[:5] == 'float', "'alpha' must be a float object."
    assert 0 < alpha < 1, "'alpha' must lie in the (0, 1) range."
    assert perm is None or perm.__class__.__name__[:3] == 'int', "'perm' must be None or an int."
    if perm is not None: st = st.nonparam
    assert test_accuracy or not test_accuracy, "'test_accuracy' must be a bool object."
    assert accuracy_trials.__class__.__name__[:3] == "int", "'accuracy_trials' must be an int object."

    # check the suitability of the factors to the spm1d t test
    assert factor.__class__.__name__ == 'DataFrame', "'factor' must be a pandas.DataFrame."
    assert factor.shape[1] == 1, "'factor' must have 1 column."
    assert factor.shape[0] == data.shape[0], "'data', and 'factor' must have equal row number."
    assert len(np.unique(factor.values)) <= 2, "'factor' must have no more than 1 or 2 distinct values."

    # understand the kind of t test to perform
    if len(np.unique(factor.values)) == 1:

        # perform a one-sample t test
        fun = st.ttest
        params = [data.values, np.mean(data.values, axis=0), roi]
    else:

        # extract the indices leading to the two samples to be compared
        idx1 = np.argwhere(factor.values.flatten() == np.unique(factor.values.flatten())[0]).flatten()
        idx2 = np.argwhere(factor.values.flatten() == np.unique(factor.values.flatten())[1]).flatten()
        assert len(idx1) == len(idx2), "The 2 components of factor must have the same number of samples."

        # check if paired t-test or 2 samples t test can be performed
        matched_idx = [i in data.iloc[idx1].index.to_numpy() for i in data.iloc[idx2].index.to_numpy()]
        assert np.all(matched_idx) or not np.any(matched_idx), "subjects are not evenly distributed between groups."

        # get the data corresponding to the two samples
        data1 = np.squeeze(data.iloc[idx1].values)
        data2 = np.squeeze(data.iloc[idx2].values)

        # setup the test
        fun = st.ttest_paired if np.all(matched_idx) else st.ttest2
        params = [data1, data2, roi]

    # prepare the parameters that need to be passed to the inference method
    ii = {'alpha': alpha}
    ii.update({} if perm is None else {'iterations': min(perm, fun(*params).nPermUnique), 'force_iterations': True})
    ii.update({'interp': False} if data.shape[1] > 1 else {})

    # generate the accuracy pool
    pool = 1 + (accuracy_trials  if test_accuracy else 0)

    # perform the inference on data
    tests = []
    for i in np.arange(pool):
        infer = fun(*params).inference(**ii)

        # store the outcome of the test
        if i == 0:
            out = infer
        tests += [__map_significance__({'A': infer})]
    
    # generate the accuracy table (for each effect)
    acc = (1 - np.mean([abs(tests[0]['A'] - i['A']) for i in tests[1:]])) if test_accuracy else None

    # get the dataframe
    df = __spm1d_df__(infer, [i for i in data.columns])
    df.insert(0, 'Accuracy', np.tile(acc, df.shape[0]))
    return infer, df



# generate the output DataFrame
def __spm1d_df__(effect, ref):
    """
    generate a pandas.DataFrame containing all the data related to the effect entered as input.

    Input:
        effect: (spm1d test object)
            an spm1d test object
        ref: (ndarray, list)
            the nodes name to be used as identifier for the start and the stop of each cluster.
    Output:
        df: (pandas.DataFrame)
            a pandas.DataFrame containing all the relevant results of the test.
    """

    # import dependancies
    from pandas import DataFrame, Series

    # build the dataframe
    df = DataFrame()
    line = DataFrame()
    try: line['Unique permutations'] = Series(effect.nPermUnique)
    except Exception: None
    try: line['permutations'] = Series(effect.nPerm)
    except Exception: None
    line['Alpha'] = Series(effect.alpha)
    line['Threshold'] = Series(effect.zstar)

    # add 0D parameters
    if len(ref) == 1:
        line[effect.STAT] = Series(effect.z)
        line['P-value'] = Series(effect.p)
        df = df.append(line)
    else:

        # manage 1D data
        line['Statistic'] = Series(effect.STAT)
        if len(effect.clusters) == 0:
            line['n'] = Series()
            line['Start'] = Series()
            line['Stop'] = Series()
            line['P-value'] = Series()
            df = df.append(line)
        else:
            for i, v in enumerate(effect.clusters):
                line['n'] = Series(int(i) + 1)
                line['Start'] = Series(ref[int(round(v.endpoints[0]))])
                line['Stop'] = Series(ref[int(round(v.endpoints[1]))])
                line['P-value'] = Series(v.P)
                df = df.append(line)

    # return the dataframe
    df.index = range(df.shape[0])
    return df



def cint(data, value, **kwargs):
    """
    computer the confidence interval of data.

    Input:
        data: (ndarray)
            a 1D array of float or int values.
        value: (float)
            a value within the (0, 100] range.

    Output:
        the confidence interval
    """

    # check data
    assert data.__class__.__name__ == "ndarray", "'data' must be a ndarray."
    assert value.__class__.__name__[:5] == "float" or value.__class__.__name__[:3] == "int", "'value' must be a scalar."
    if not 'axis' in kwargs:
        axis = -1
    else:
        axis=kwargs['axis']

    # import dependancies
    from scipy.stats import sem, t
    from numpy import mean

    # compute the interval
    avg = mean(data, axis)
    se = sem(data, axis)
    p = t.ppf((1 + value / 100) / 2, data.shape[axis] - 1)
    return [avg - p * se, avg + p * se]



# %% SIDAK CORRECTION FOR MULTIPLE COMPARISONS
def sidak_alpha(alpha, n_test):
    '''
    return the corrected alpha for the given number of tests

    Input
        alpha:  (float)
                alpha value.

        n_test: (int)
                the numeber of tests

    Output
        corr:   (float)
                tha alpha value corrected using the Bonferroni approach
    '''

    # import the necessary packages
    import numpy as np

    # check the entered parameters
    S = '"alpha" must be a single float number.'
    assert len(np.array([alpha]).flatten()) == 1, S
    S = '"alpha" must be in the [0, 1] range'
    assert alpha >= 0 and alpha <= 1, S
    S = '"n_test" must be an integer.'
    assert len(np.array([n_test]).flatten()) == 1, S
    assert np.floor(n_test) == n_test, S

    # return the corrected alpha
    return 1 - ((1 - alpha) ** (1 / n_test))


# %% STATISTICAL NON PARAMETRIC CLUSTER
class SnPMCluster():
    '''
    Generate a Statistical Permutation Cluster using a Statistical Map resulting from a SPM object.

    Input
         Z   :   (1D array)
                 The statistical map.

        ZZ  :   (nD array)
                the statistical maps corresponding to the permuted copies of the original data. This array is
                sorted having each row corresponding to one permutation.

        location:(1D array)
                an array of int defining the regions of Z corresponding to the cluster.

        threshold:(float)
                the critical threshold defining the level of significance.

        alpha:  (float)
                the level of significance.

    Output
        X   :   (1D array)
                the extrema of the statistical map that allowed for the detection of the clusters.

        N   :   (int)
                the extension of the cluster in points number

        thresh: (float)
                the critical threshold defining the level of significance.

        A   :   (float)
                the area under the curve of the statistical map within range and above the critical threshold.

        PDF  :  (1D array)
                the omologous of P calculated for each permuted copies of the original data. It represents the
                probability density function for the calculation of the p value.

        P   :   (float)
                the p value associated to the cluster.

    Procedure
        1)  Get the statistical map region corresponding to the given location (region).

        2)  Build the probability density function (PDF) over which calculate the p values. This will be accomplished
            calculating the area under the curve (but above the critical threshold) of the location provided for each
            permuted copy of the data.

        3)  Get the p value as the probability that the area of the current test is lower the the area resulting from
            the permuted copies.

    References
        Nichols TE, Holmes AP. Nonparametric permutation tests for functional neuroimaging: A primer with examples. Hum
            Brain Mapp. 2002;15(1):1–25.

        Pataky TC, Vanrenterghem J, Robinson MA. Zero- vs. one- dimensional, parametric vs. nonparametric, and
            confidence interval vs. hypothesis testing procedures in one-dimensional biomechanical trajectory analysis.
            J Biomech. 2015;48(7):1277–85.
    '''

    # constructor
    def __init__(self, Z, ZZ, location, threshold, alpha, par):

        # import the required packages
        import numpy as np

        # assess Z
        assert Z.ndim == 1, 'Z must be a 1D array.'
        self.Z = np.array(Z).flatten()

        # get the nodes
        Q = len(Z)

        # assess ZZ
        assert ZZ.shape[1] == Q, 'ZZ must have ' + str(Q) + ' columns.'

        # assess the location
        ST = 'location must stay within the [0, ' + str(Q) + ') range.'
        assert np.min(location) >= 0 and np.max(location) < Q, ST

        # add the range and the extent of the cluster
        self.X = np.array([np.sort(location)]).flatten().astype(int)
        self.N = np.sum([i for i in np.diff(self.X) if i == 1])

        # assess the threshold
        assert threshold > 0, 'threshold must be a float abouve 0'
        assert len(np.array([threshold])) == 1, 'threshold must be a scalar.'
        self.thresh = threshold

        # assess alpha
        assert alpha > 0 and alpha < 1, 'alpha must lie in the (0, 1) range.'
        assert len(np.array([alpha])) == 1, 'alpha must be a scalar.'

        # get A
        self.A = self.__AUC__((self.Z[self.X], self.thresh))

        # get the PDF
        self.PDF = [(i[self.X], self.thresh) for i in ZZ]
        self.PDF = np.array([self.__AUC__(i) for i in self.PDF] if par is None else par.map(self.__AUC__, self.PDF))

        # get the p value
        self.P = max((self.PDF > self.A).mean(), 1. / ZZ.shape[0])

    # parallel computation of the area under the curve
    def __AUC__(self, T):
        '''
        It returns the area under the curve of one array over a second one.

        Input
            T   :   (tuple)
                    a tuple containing two values:
                        A   :   (1D array)
                                an array with data

                        B   :   (1D array or float)
                                an array or a scalar value that is subtracted from A before computing the area under
                                the curve.

        Output
            A   :   (float)
                    the area under the curve of A over B using the trapezoid method.
        '''

        # import the necessary packages
        import numpy as np

        # check the tuple data
        A = np.atleast_1d(T[0])
        assert A.ndim == 1, '"A" must be a 1D array.'
        B = np.atleast_1d(T[1])
        assert B.ndim == 1, '"B" must be a 1D array or a float value.'

        # return the area under the curve
        return A - B if len(A) == 1 else np.trapz(A - B)

    # return the data of the cluster as pandas DataFrame line
    def to_df(self):
        ''' convert the data of the cluster as pandas DataFrame line'''

        # import the necessary packages
        from pandas import DataFrame, Series

        # create the output dataframe
        L = DataFrame()
        L['SPC_auc'] = Series(self.A)
        L['SPC_thresh'] = Series(self.threshold)
        L['SPC_p'] = Series(self.P)
        L['SPC_n'] = Series(self.extent)
        L['SPC_from'] = Series(self.X[0])
        L['SPC_to'] = Series(self.X[-1])

        # return the dataframe line
        return L


# %% 1D STATISTICAL PERMUTATION MAP:
class SnPMEffect():
    '''
    Create a 1D Statistical Permutation Map according to a given statistical test. Despite this function can be
    used also by itself, it is not designed to be used directly. Indeed, it is called during the initialization of
    a SnPM class object to analyse each effect of the full factorial design.

    Input
        data: (pandas DataFrame)
            a (J-by-Q) pandas DataFrame where J is the number of samples and Q the number of nodes.

        factors: (DataFrame)
            a (J-by-N) pandas DataFrame where N is the number of factors.

        test: (method)
            the function to be used to calculate the statistical test which will be used to generate the
            statistical maps. The test function must accept at least the parameters data and factors and must
            return a test statistic for each node Q.

        perm: (int, default=1e5)
            the number of permutations to be used. If N is higher than the possible number of permutations, all of
            them are performed. Otherwise, the selected number of permutations are randomly sampled.

        alpha: (float, default=0.05)
            the level of significance to be used.

        twotailed: (boolean, default=True)
            if True (default) the critical threshold is calculated according to a two-tailed approach. A one-tailed
            approach is used otherwise.

        circular: (boolean, default=True)
            if True, the starting and endpoints of the map are considered contiguous. Therefore, if a cluster starts at
            the beginning of the map and the same cluster (or another) ends at the end of the map, they are merged
            together. This affects the p-value calculation for the cluster.

        multi_index: (str)
            During its construction, SnPM objects check for unique index-factors combinations. If multiple copies of the
            same index are found for one or more factor combinations, this attribute determines how to manage them:
                "mean": the factors are averaged.
                "median": the median is given.


        args, kwargs: (any)
            additional arguments to be passed to "test".

        Output:
            J: (int)
                the number of samples.

            Q: (int)
                the number of nodes.

            test: (method)
                the test used to generate the statistical maps.

            perm_max: (int)
                the maximum number of permutations allowed by the data.

            perm: 2D array
                a 2D array where each row contains the indices ensuring nperm unique permutations of data.

            alpha: (float)
                the critical p-value.

            twotailed: (bool)
                is the p-value considered from a 1 or 2-tailed distribution?

            circular: (bool)
                is the statistical map being considered as circular?

            multi_index: (str)
                the behaviour used to manage multiple occurrences of the same index within the same factor combination.

            Z: (1D array)
                the statistical map

            threshold: (float)
                the critical threshold in the map corresponding to the given level of alpha.

            PDF : (1D array)
                the probability density function used to calculate the critical threshold.

            clusters:(list)
                a list of SPC objects defining each cluster.

        Procedure:
            1)  Get the statistical map corresponding to the current test.

            2)  Build the probability density function (PDF) over which extract the critical threshold for the cluster
                detection phase. During this step get the max statistical test value obtained after having permuted the
                data.

            3)  Get the critical threshold as the n-th percentile of the PDF according to the defined level of
                significance.

            4) Define the clusters as the regions in the statistical map above the level of significance.

            5) Get the clusters (look at the SPC class description for further information).

            6) Keep those clusters resulting in significan p-values.

        References
            Nichols TE, Holmes AP. Nonparametric permutation tests for functional neuroimaging: A primer with examples.
                Hum Brain Mapp. 2002;15(1):1–25.
            Pataky TC, Vanrenterghem J, Robinson MA. Zero- vs. one- dimensional, parametric vs. nonparametric, and
                confidence interval vs. hypothesis testing procedures in one-dimensional biomechanical trajectory
                analysis. J Biomech. 2015;48(7):1277–85.
        '''


    # constructor
    def __init__(self, data, factors, test, perm, alpha, twotailed, circular, multi_index, *args, **kwargs):

        # import the required packages
        import numpy as np
        from pandas import DataFrame, concat

        # check data
        assert data.__class__.__name__ == "DataFrame", "'data' must be a 'pandas.DataFrame' object."
        (J, Q) = data.shape
        self.J = J
        self.Q = Q

        # check factors
        assert factors.__class__.__name__  == 'DataFrame', '"factors" must be a (pandas) DataFrame.'
        assert factors.shape[0] == J, '"factors" must have ' + str(J) + ' rows.'
        assert np.all([v == data.index[i] for i, v in enumerate(factors.index)]), "'factors' and 'data' index differ."

        # store the test as is (no check will be performed as the function  itself must check the entered parameters)
        self.test = test

        # check the permutation number
        assert perm.__class__.__name__[:3] == 'int', '"perm" must be an int value.'

        # check alpha
        assert alpha.__class__.name__[:5] == "float", "'alpha' must be a float."
        assert 0 < alpha < 1, "'alpha' must lie in the (0, 1) range."
        self.alpha = alpha

        # check the two_tailed option
        assert twotailed or not twotailed, '"twotailed" must be a boolean.'
        self.twotailed = twotailed

        # get the critical p-value
        self.critical_p = alpha * (0.5 if twotailed else 1)

        # check the circular option
        assert circular or not circular, '"circular" must be a boolean.'
        self.circular = circular

        # check multi_index
        assert multi_index in ['mean', 'median'], "'multi_index' must be 'mean' or 'median'."
        self.multi_index = multi_index

        # get the reduced data (i.e. 1 index occurrence per factor combination)
        combs = concat([DataFrame(factors.index.to_numpy(), index=factors.index, columns=['Counts']), factors], axis=1)
        reduced_data = concat([data, combs], axis=1)
        if multi_index == 'mean': reduced_data = combs.groupby(combs.columns, as_index=False).mean(axis=0)
        elif multi_index == 'median': reduced_data = combs.groupby(combs.columns, as_index=False).median(axis=0)
        reduced_data.index = reduced_data['Counts'].values
        reduced_factors = reduced_data[factors.columns]
        reduced_data = reduced_data[data.columns].values

        # get the maximum number of permutations
        # it is: perm_max = J! / (n1! * n2! * ...* nk!), with k = number of unique factor combinations and n the number of
        # elements on each combination.
        factorial = lambda n: int(np.prod(np.arange(n) + 1))
        unique_fctrs = DataFrame(reduced_factors.index.to_numpy(), index=reduced_factors.index, columns=['Counts'])
        unique_fctrs = concat([unique_fctrs, reduced_factors], axis=1)
        self.perm_max = factorial(factors.shape[0])
        self.perm_max /= np.prod([factorial(i) for i in unique_fctrs['Counts'].values])
        self.perm_max = int(self.perm_max)

        # get the nperm permutations
        ref = np.concat([np.tile(i, v) for i, v in enumerate(unique_fctrs['counts'].values)], axis=0).flatten()
        idx= np.atleast_2d(reduced_factors.index.to_numpy())
        n = np.min([self.perm_max, perm])
        while self.perm.shape[0] < n:
            np.random.seed()
            k = np.atleast_2d([np.random.permutation(idx) for i in np.arange(n - self.perm.shape[0])])
            self.perm = k if self.perm.shape[0] == 0 else np.vstack((self.perm, k))
            self.perm = self.perm[np.unique(np.atleast_2d([ref[i] for i in self.perm]), axis=0, return_index=True)[1]]

        # define the statistical map
        self.Z = self.test(reduced_data, reduced_factors, *args, **kwargs)

        # create the statistical maps for each permuted copy
        ZZ = np.atleast_2d([test(reduced_data[i], reduced_factors, *args, **kwargs) for i in self.perm])

        # get the PDF
        self.PDF = np.max(ZZ, 1)

        # get the critical threshold
        self.thresh = np.percentile(self.PDF, self.critical_p)

        # get the clusters
        C = np.argwhere(abs(self.Z) > self.thresh).flatten()
        if len(C) > 0:

            # look for cluster breaks
            X = np.argwhere(np.diff(C) > 1).flatten()

            # generate clusters range
            X = np.append(C[X], C[X + 1]) if len(X) > 0 else []
            X = np.sort(np.concatenate([np.array([C[0]]), X, np.array([C[-1]])]))

            # define each cluster
            i = 0
            C = []
            while i < len(X):
                C += [np.arange(X[i], X[i + 1] + 1)]
                i += 2

            # adjust for circularity
            if len(C) > 1 and circular:
                if C[0][0] == 0 and C[-1][-1] == len(self.Z) - 1:
                    C[0] = np.unique(np.concatenate((C[0], C[-1])))
                    C = C[:-1]

            # get the clusters
            C = [SnPMCluster(self.Z, ZZ, i, self.thresh, alpha) for i in C]

            # remove those clusters having P value above alpha
            self.clusters = [i for i in C if i.P < self.critical_p]

        # generate an empty clusters list
        else:
            self.clusters = []

    # return the dataframe summarizing each significant cluster within the map
    def to_df(self):
        '''convert the data of the cluster as pandas DataFrame line'''

        # import the necessary packages
        import pandas as pd
        import numpy as np

        # generate the output dataframe
        D = pd.DataFrame()
        D['SPM_perm_max'] = pd.Series(self.perm_max)
        D['SPM_perm'] = pd.Series(self.perm)
        D['SPM_thresh'] = pd.Series(self.thresh)
        U = [pd.concat((D, cl.to_df()), 1) for cl in self.clusters]
        U = pd.concat(U, 0)
        U.index = np.arange(U.shape[0])

        # return the dataframe
        return U


# %% PERMUTATION TEST ANALYSIS
class SnPM_():
    '''
    Create a 1D Statistical Permutation Map according to a given statistical test.
    '''

    # constructor
    def __init__(self, data, factors, idx, test, perm=1e5, alpha=0.05, twotailed=True, circular=True,
                 multi_index='mean', *args, **kwargs):
        '''
        Input
            data: (J-by-Q 2D array)
                The dataset. It is casted to float values. J is the number of distinct acquisitions, while Q is the
                number of nodes for each acquisiiton.

            factors: (a pandas DataFrame with J rows)
                The parameters used to define the groups being compared. All the parameters are casted to be string.
                J is the number of distinct acquisitions, while N is the number of factors.

            idx: (ndarray or None)
                The index reflecting a repeated measurement design. It must have dimension 1 and length equal to J.
                If None a sequence of increasing integers is used, thus assuming non-repeated measurements.

            test: (method)
                the function to be used to calculate the statistical test which will be used to generate the
                statistical maps. The test function must accept at least the parameters data and factors and must
                return a test statistic for each node Q.

            perm: (int, default=1e5)
                the number of permutations to be used. If N is higher than the possible number of permutations, all
                of them are performed. Otherwise, N permutations are randomly used.

            alpha: (float, default=0.05)
                the level of significance to be used.

            twotailed: (boolean, default=True)
                if True (default) the critical threshold is calculated according to a two-tailed approach. A
                one-tailed approach is used otherwise.

            circular: (boolean, default=True)
                if True, the starting and endpoints of the map are considered contiguous. Therefore, if a cluster
                starts at the beginning of the map and the same cluster (or another) ends at the end of the map,
                they are merged together. This affects the p-value calculation for the cluster.

            multi_index: (str)
                During its construction, SnPM objects check for unique index-factors combinations. If multiple
                copies of the same index are found for one or more factor combinations, this attribute determines
                how to manage them:
                    "mean": the factors are averaged.
                    "median": the median is given.

        Output
            a SnPM class object returning:

                data: (pandas.DataFrame)
                    the data used in the analysis.

                J: (int)
                    the number of distinct acquisitions.

                Q: (int)
                    the number of nodes.

                alpha: (float)
                    the level of significance.

                test: (function)
                    the function used to get the statistical map

                twotailed: (boolean)
                    returns if the statistics have been calculated using a two-tailed approach.

                circular: (boolean)
                    returns if the statistics have been calculated using a circular approach (i.e. the end of the
                    data correspond to the start of a new cycle).

                effects: (dict)
                    a dict object containing all the main and interaction effects resulting from the factors
                    provided. Each effect will be a SnPM object. In case of interaction effects, the ":" symbol
                    will separate the single factors being considered for interaction.

                multi_index: (str)
                    the behaviour in case of multiple copies of the same indices for the same factor combination.
                    This parameter is passed to the SnPM call to calculate the statistical effects.
            '''

        # import the required packages
        import numpy as np
        import pandas as pd
        from itertools import combinations

        # check data
        assert data.__class__.__name__ in ["DataFrame"], "'data' must be a (pandas) DataFrame."
        data = data.astype(float)
        (J, Q) = data.shape
        self.samples = J
        self.nodes = Q
        data.index = np.arange(J)

        # check the factors object
        assert factors.__class__.__name__ in ['DataFrame'], '"factors" must be a (pandas) DataFrame.'
        fctrs = pd.DataFrame(factors)  # ensure factors is a pandas DataFrame
        assert fctrs.shape[0] == J, '"factors" must have ' + str(J) + ' rows.'
        fctrs.index = np.arange(J)

        # check idx
        assert idx.__class__.__name__ in ["ndarray", 'list'], "'idx' must be a 'ndarray' or a list."
        idx = np.array([idx]).flatten().astype(str)
        assert len(idx) == J, "'idx' must contain " + str(J) + " samples."

        # check the permutation number
        assert perm.__class__.__name__[:3] == 'int', '"perm" must be an int value.'

        # get the maximum number of permutations
        # it is: perm_max = J! / (n1! * n2! * ...* nk!), with k = number of unique factor combinations and n the
        # number of elements on each combination.
        factorial = lambda n: int(np.prod(np.arange(n) + 1))
        unique_fctrs = fctrs.groupby(fctrs.columns.to_numpy()).count()
        self.perm_max = int(np.round(factorial(J) / np.prod([factorial(i) for i in unique_fctrs[['Counts']]])))

        # get the nperm permutations
        combs = np.atleast_2d([])
        perms = []
        np.random.seed()
        n = np.min([perm, self.perm_max])
        count = 0
        test = ["-".join(i) for i in np.unique(np.concatenate([fctrs.values, np.atleast_2d(idx).T], axis=1), axis=0)]
        while len(perms) < n:
            count += 1
            prm = np.random.permutation(np.arange(J))
            cmb = np.unique(np.concatenate([fctrs.values, np.atleast_2d(idx).T], axis=1), axis=0)
            cmb = np.atleast_2d(["-".join(i) for i in cmb])
            if len(combs) == 0 or ~np.any([np.array_equiv(cmb[0], i) for i in combs]):
                combs = cmb if len(combs) == 0 else np.vstack((combs, cmb))
                perms += [prm]
            print('\rPermutations found: {0.1f} % in '.format(100 * len(perms) / n) + str(count) + " cycles.", end="")
        print("")

        # store the test as is (no check will be performed as the function  itself must check the entered parameters)
        self.test = test

        # check the two_tailed option
        assert twotailed or not twotailed, '"twotailed" must be a boolean.'
        self.twotailed = twotailed

        # check the circular option
        assert circular or not circular, '"circular" must be a boolean.'
        self.circular = circular

        # check the level of significance
        assert alpha.__class__.__name__[:5] == 'float', '"alpha" must be a float.'
        assert alpha > 0 and alpha < 1, '"alpha" must lie in the (0, 1) range.'
        self.alpha = alpha

        # check multi_index
        assert multi_index in ['mean', 'median'], "'multi_index' must be 'mean' or 'median'."
        self.multi_index = multi_index

        # generate the SnPM stats effect by effect
        self.effects = {
            ":".join(i): SnPMEffect(data, fctrs[i], test, perm, alpha, twotailed, circular, multi_index, *args,
                                    **kwargs)
            for i in [i for n in np.arange(factors.shape[1]) + 1 for i in combinations(factors.columns, n)]
            }

    def to_df(self):
        '''Return a dataframe summarizing the class features.'''

        # import the necessary packages
        import pandas as pd
        import numpy as np

        # generate the general data line
        L = pd.DataFrame({'PAN_samples': self.J, 'PAN_nodes': self.Q, 'PAN_alpha': self.alpha,
                          'PAN_two_tailed': self.twotailed, 'PAN_circular': self.circular})
        D = [pd.concat((L, pd.DataFrame([i], columns=['Effect']), self.effects[i].to_df()), 1) for i in self.effects]
        D = pd.concat(D, 0)
        D.index = np.arange(D.shape[0])

        return D



def snpm_anova(df, nodes, between=None, within=None, sub=None, perm=1e5, alpha=0.05, circular=True, two_tailed=True):
    """
    perform a statistical non-parametric mapping anova test.

    Input:
        df: (pandas.DataFrame)
            a dataframe containing all the data.

        nodes: (list, ndarray)
            a list containing the column names in df corresponding to the nodes of the map.

        between: (list, ndarray, None)
            a list containing the column names in df corresponding to the between factors.

        within: (list, ndarray, None)
            a list containing the column names in df corresponding to the within factors.

        sub: (str, None)
            a the subjects column name.

        alpha: (float)
            the level of significance.

        perm: (int, None)
            the number of permutations to be used for the non-parametric anova. If None, parametric analysis is
            performed, otherwhise non-parametric analysis with n=perm permutations is computed. If perm=-1, all the
            possible permutations are used.

    Output:
        summary: (dict)
            a dict containing:
                effects: (dict)
                    a dict with the stats referring to each effect.
                df: (pandas.DataFrame)
                    a pandas.DataFrame summarizing the results of the analysis.
    """

    # import the required packages
    import numpy as np
    import pandas as pd
    from itertools import combinations

    # check entered parameters
    assert df.__class__.__name__ in ["DataFrame"], "'df' must be a (pandas) DataFrame."
    assert nodes.__class__.__name__ in ['list', 'ndarray'], '"nodes" must be a list or ndarray.'
    assert within.__class__.__name__ in ['list', 'ndarray', 'NoneType'], '"within" must be a list or ndarray.'
    assert between.__class__.__name__ in ['list', 'ndarray', 'NoneType'], '"between" must be a list or ndarray.'
    assert idx.__class__.__name__ in ['list', 'ndarray', 'NoneType'], '"between" must be a list or ndarray.'

