


# IMPORTS



from .stats import *
import numpy as np
import pandas as pd
import itertools as it
import scipy.signal as ss
import scipy.interpolate as si
from bokeh.plotting import *
from bokeh.layouts import *
from bokeh.models import *



# GENERAL OPTIONS
fig_size = 300  # pixels



# METHODS



def digitize(x, rule="fd"):
    """
    return a digitized version of x where each value is linked to a bin (i.e an int value) according to a specific
    rule.

    Input:
        x:      (nD array)
                a ndarray that has to be digitized.

        rule:   (str)
                the method to be used for digitizing x.
                Currently, only the Freedmain-Diaconis rule (fd) is implemented, which stratify the x amplitude into
                intervals having width:
                                                         IQR(x)
                                        width = 2 * ---------------
                                                    len(x) ** (1/3)

    Output:
        
        d:      (ndarray)
                an array with the same shape of x but of dtype "int" where each element denotes the digitized amplitude
                of the corresponding value in x.

    References:
        Freedman D, Diaconis P. (1981) On the histogram as a density estimator:L 2 theory.
            Z. Wahrscheinlichkeitstheorie verw Gebiete 57: 453–476. doi: 10.1007/BF01025868
    """

    # scale x
    z = (x - np.min(x)) / (np.max(x) - np.min(x))

    # get the width
    if rule == "fd":    w = 2 * IQR(z) / (len(z) ** (1 / 3))

    # only "fd" is currently supported
    else:    raise AttributeError("Only Freedman-Diaconis rule is implemented. Please run again using 'fd' as rule.")

    # get the number of intervals
    n = int(np.floor(1 / w)) + 1

    # digitize z
    d = np.zeros(z.shape)
    for i in (np.arange(n) + 1):
        d[np.argwhere((x >= (i - 1) * w) & (x < i * w)).flatten()] = i - 1
    return d




def moving_average_filter(y, n=1, offset=0, pad_style="mirror", plot=False):
    """
    apply a moving average filter with the given order to y. The only padding style currently supported is "mirror"
    which specularly extends the signal at each end to the required amount of samples necessary to preserve the signal
    length output.
    
    Input:
        y:          (1D array)
                    a 1D array signal to be filtered

        n:          (int)
                    the order of the filter. It must be odd.

        offset:     (float)
                    should the window be centered around the current sample?.
                    The offset value should range between -1 and 1. with -1 meaning that at each i-th sample the 
                    window of length n over which the mean will be used as filter output for i will be calculated
                    having i as starting value. Conversely, 1 means that for the same i-th sample, n-1 samples
                    behind i will be used to calculate the mean. An offset of value 0 will consider i centered
                    within the filtering window.

        pad_style:  (str)
                    currently "mirror" is the only padding style supported. This style means that in order to
                    preserve the length of the output signal, the raw signal is mirrored at each end before
                    applying the filter.

        plot:       (bool)
                    if True a bokeh figure representing the output of the filter.

    Output:
        z:          (1D array)
                    The filtered signal.

        p:          (bokeh.figure, optional)
                    if plot is True, the a figure representing the output of the filter.
    """

    # check n is odd
    assert n % 2 == 1, "'n' must be odd."
    assert n > 0, "'n' must be higher than 0."

    # ensure n is an int
    n = int(n)

    # check offset is within the [-1, 1] range
    assert (offset >= -1) & (offset <= 1), "'offset' must be a scalar within the [-1, 1] range."

    # ensure pad_style is supported
    assert pad_style == "mirror", "'mirror' is the only pad_style supported."

    # get the number of padding samples necessary for each end
    left_pad = int(abs(np.round((n - 1) / 2 * (offset - 1))))
    right_pad = n - 1 - left_pad

    # get the padded signal
    if pad_style == "mirror":

        # assuming that left_pad and right_pad equal 2 and 3 respectively, we want:
        #           y = [ABCDEFGHI]     -->       y_padded = [CB] [ABCDEFGHI] [HGF]
        y_pad = np.concatenate([y[1:(left_pad + 1)][::-1], y, y[-(right_pad + 1):-1][::-1]]).flatten()

    # to reduce the computation time rather than calculating the mean value corresponding to each sample of the output
    # signal, we calculate the filter from the cumulative sum. This greatly reduces the number of operations, thus the
    # computational time, especially if the filter order is high.
    y_c = np.cumsum(y_pad)  # cumulative sum
    z = (y_c[(n - 1):] - np.append([0], y_c[:(len(y) - 1)])) / n  # filtered signal
    
    # check if the plot must be provided
    if not plot:
        return z

    # generate the output plot figure
    p = figure(width=fig_size, height=fig_size, title="moving average filter")
                
    # edit the axes labels
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "Y"

    # plot the True data
    x = np.arange(len(y))
    p.scatter(x, y, size=2, color="navy", alpha=0.5, marker="circle", legend_label="Raw")
    p.line(x, y, line_width=1, color="navy", alpha=0.5, line_dash=(3, 3), legend_label="Raw")
    
    # plot the interpolated data
    p.scatter(x, z, size=2, color="darkred", alpha=0.5, marker="circle", legend_label="Filtered")
    p.line(x, z, line_width=1, color="darkred", alpha=0.5, line_dash=(3, 3), legend_label="Filtered")

    # set the legend position
    p.legend.location = "top_right"
    p.legend.title = "Legend"
    p.legend.click_policy = "hide"
    
    # edit the grids
    p.xgrid.grid_line_alpha=0.3
    p.ygrid.grid_line_alpha=0.3
    p.xgrid.grid_line_dash=[5, 5]
    p.ygrid.grid_line_dash=[5, 5]
    
    # return all
    return z, p



def cubic_spline_interpolation(y, x_old=None, x_new=None, n=None, plot=False, **kwargs):
    """
    Get the cubic spline interpolation of y.

    Input:
        y:      (1D array)
                the data to be interpolated.

        n:      (int)
                the number of points for the interpolation. It is ignored if x_new is provided.

        x_old:  (1D array)
                the x coordinates corresponding to y. If not provided, it is set as an increasing set of int with
                length equal to Y.
        
        x_new:  (1D array)
                the x coordinates corresponding to y. If provided, n is ignored.

        plot:   (bool)
                if True, a bokeh.Figure is provided.

        kwargs: additional parameters passed to the scipy.interpolate.CubicSpline function

    Output:
        ys:     (1D array)
                the interpolated y axis
        
        p:      (bokeh.Figure, optional)
                a figure representing the fit of the model.
    """

    # get x_old if not provided
    if x_old is None:
        x_old = np.arange(len(y))

    # get the cubic-spline object
    cs = si.CubicSpline(x_old, y)

    # get x_new if not provided
    if x_new is None:
        x_new = np.linspace(np.min(x_old), np.max(x_old), n)

    # get the interpolated y
    ys = cs(x_new)

    # return ys if plot is None
    if not plot:
        return ys

    # generate the output plot figure
    p = figure(width=fig_size, height=fig_size, title="Cubic-spline interpolation")
                
    # edit the axes labels
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "Y"

    # plot the True data
    p.scatter(x_old, y, size=2, color="navy", alpha=0.5, marker="circle", legend_label="True")
    p.line(x_old, y, line_width=1, color="navy", alpha=0.5, line_dash=(3, 3), legend_label="True")
    
    # plot the interpolated data
    p.scatter(x_new, ys, size=2, color="darkred", alpha=0.5, marker="circle", legend_label="Interpolated")
    p.line(x_new, ys, line_width=1, color="darkred", alpha=0.5, line_dash=(3, 3), legend_label="Interpolated")

    # set the legend position
    p.legend.location = "top_right"
    p.legend.title = "Legend"
    p.legend.click_policy = "hide"
    
    # edit the grids
    p.xgrid.grid_line_alpha=0.3
    p.ygrid.grid_line_alpha=0.3
    p.xgrid.grid_line_dash=[5, 5]
    p.ygrid.grid_line_dash=[5, 5]
    
    # return all
    return ys, p



def winter_residuals(x, fs, f_num=1000, f_max=None, segments=2, min_samples=2, filt_fun=None, filt_opt=None,
                     plot=False):
    """
    Perform Winter's residual analysis of the entered data.

    Input:
        x:  (numpy 1D array)
            the signal to be investigated
        fs: (float)
            the sampling frequency of the signal.
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
        cutoffs: (float)
            a pandas.DataFrame with one value per column defining the optimal cutoff frequency.
        SSEs: (pandas.DataFrame)
            a pandas.DataFrame with the selected frequencies as index and the Sum of Squared Residuals as columns.
        fig: (bokeh.figure)
            a figure is provided if plot is True

    Procedure:
        the signal is filtered over a range of frequencies and the sum of squared residuals (SSE) against the original
        signal is computer for each tested cut-off frequency. Next, a series of fitting lines are used to estimate
        the optimal disruption point defining the cut-off frequency optimally discriminating between noise and good
        quality signal.

    References:
        Winter DA. Biomechanics and Motor Control of Human Movement. Fourth Ed. Hoboken,
            New Jersey: John Wiley & Sons Inc; 2009.
        Lerman PM. Fitting Segmented Regression Models by Grid Search. Appl Stat. 1980;29(1):77.
    """

    # get the frequency span
    freqs = np.linspace(0, (fs * 0.25) if f_max is None else f_max, f_num + 2)[1:-1]
       
    # get the filter function options
    if filt_fun is None:
        filt_fun = butt_filt
        filt_opt = {'order': 4, 'sampling_frequency': fs, 'type': 'lowpass', 'phase_corrected': True, 'plot': False}
    else:
        if filt_opt is None:
            filt_opt = {}

    # get the SSEs
    Q = np.array([np.sum((x - filt_fun(x, cutoffs=i, **filt_opt)) ** 2) for i in freqs])
    
    # get the optimal crossing over point that separates the S regression lines best fitting the residuals data.
    Z = crossovers(Q, None, segments, min_samples, False)

    # get the intercept of the second line (i.e. the most flat one)
    fit = np.polyfit(freqs[Z[0][-1]:], Q[Z[0][-1]:], 1)

    # get the optimal cutoff
    opt = freqs[np.argmin(abs(Q - fit[-1]))]

    # get the cutoff frequency
    if not plot:
        return opt, pd.DataFrame({'SSE': Q}, index=freqs)

    # generate the output plot figure
    p = figure(width=fig_size, height=fig_size, title="Winter residual analysis")
                
    # edit the axes labels
    p.xaxis.axis_label = "Frequency (Hz)"
    p.yaxis.axis_label = "Sum of Squared Errors"

    # plot the data
    p.scatter(freqs, Q, size=4, color="navy", alpha=0.5, marker="circle", legend_label="SSE")
    p.line(freqs, Q, line_width=1, color="navy", alpha=0.5, line_dash=(3, 3), legend_label="SSE")
    p.line([0, np.max(freqs)], np.polyval(fit, [0, np.max(freqs)]), line_dash=(5, 2), line_width=1,
           color="darkred", legend_label="Fitting line", line_alpha=0.5)
    p.line([0, np.max(freqs)], [fit[-1], fit[-1]], line_dash=(2, 5), line_width=1, color="darkgreen",
           legend_label="Intercept", line_alpha=0.5)
    src = ColumnDataSource(data={'x': [opt], 'y': [Q[np.argmin(abs(Q - fit[-1]))]],
                                 't': ["Optimal cutoff = {:0.1f} Hz".format(opt)]})
    p.scatter('x', 'y', size=8, color="orange", alpha=0.8, marker="circle", source=src)
    p.add_layout(LabelSet(x='x', y='y', text='t', source=src, x_offset=5, y_offset=5, border_line_color='orange',
                          border_line_alpha=0.2, background_fill_color='orange', background_fill_alpha=0.2,
                          border_line_width=1, level='glyph', render_mode='canvas'))

    # set the legend position
    p.legend.location = "top_right"
    p.legend.title = "Legend"
    
    # edit the grids
    p.xgrid.grid_line_alpha=0.3
    p.ygrid.grid_line_alpha=0.3
    p.xgrid.grid_line_dash=[5, 5]
    p.ygrid.grid_line_dash=[5, 5]
    
    # return all
    return opt, pd.DataFrame({'SSE': Q}, index=freqs), p



def crossovers(y, x=None, n_segments=2, min_samples=5, plot=False):
    """
    Detect the position of the crossing over points between K regression lines used to best fit the data.

    Procedure:
        1)  Get all the segments combinations made possible by the given number of crossover points.
        2)  For each combination, calculate the regression lines corresponding to each segment.
        3)  For each segment calculate the residuals between the calculated regression line and the effective data.
        5)  Once the sum of the residuals have been calculated for each combination, sort them by residuals amplitude.

    References
        Lerman PM.
            Fitting Segmented Regression Models by Grid Search. Appl Stat. 1980;29(1):77.

    Input:
        Y: (ndarray)
            The data to be fitted.
        X: (ndarray or None)
            The x-axis data. If not provided it will be set as range(N) with N equal to the length of Y.
        K: (int)
            The number of regression lines to be used to best fit Y.
        samples: (int)
            The minimum number of points to be used to calculate the regression lines.
        plot: (bool)
            If True a plot showing the crossing over points and the calculated regression lines is generated.

    Output:
        An ordered array of indices where the columns reflect the position (in sample points) of the crossing overs
        while the rows shows the best fitting options from the best to the worst.
    """

    # get the residuals calculating formula
    def SSEs(x, y, s):

        # get the coordinates
        C = [np.arange(s[i], s[i + 1] + 1) for i in np.arange(len(s) - 1)]
        
        # get the fitting parameters for each interval
        Z = [np.polyfit(x[i], y[i], 1) for i in C]

        # get the regression lines for each interval
        V = [np.polyval(v, x[C[i]]) for i, v in enumerate(Z)]

        # get the sum of squared residuals
        return np.sum([np.sum((y[C[i]] - v) ** 2) for i, v in enumerate(V)])
    
    # get the X axis
    if x is None:
        x = np.arange(len(y))

    # get all the possible combinations
    J = [j for j in it.product(*[np.arange(min_samples * i, len(y) - min_samples * (n_segments - i))
                                 for i in np.arange(1, n_segments)])]

    # remove those combinations having segments shorter than "samples"
    J = [i for i in J if np.all(np.diff(i) >= min_samples)]

    # generate the crossovers matrix
    J = np.hstack((np.zeros((len(J), 1)), np.atleast_2d(J), np.ones((len(J), 1)) * len(y) - 1)).astype(int)

    # calculate the residuals for each combination
    R = np.array([SSEs(x, y, i) for i in J])

    # sort the residuals
    T = np.argsort(R)

    # get the optimal crossovers order
    O = x[J[T, 1:-1]]

    # return the crossovers
    if not plot:
        return O
    
    # generate the output plot figure
    p = figure(width=fig_size, height=fig_size, title="Lerman fitting regression lines")
                
    # edit the axes labels
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "Y"

    # plot the data
    p.scatter(x, y, size=4, color="navy", alpha=0.5, marker="circle", legend_label="data")
    p.line(x, y, line_width=2, color="navy", alpha=0.5, line_dash=(3, 3), legend_label="data")
    
    # plot the optimal fitting lines
    for i in np.arange(n_segments):

        # get the segment
        seg_ix = np.argwhere((x >= x[J[T[0], i]]) & (x <= x[J[T[0], i + 1]])).flatten()
        seg_x = x[seg_ix]
        seg_y = y[seg_ix]

        # get the fitting line
        seg_z = np.polyval(np.polyfit(seg_x, seg_y, 1), seg_x)

        # plot the line
        p.line(seg_x, seg_z, line_dash=(5, 2), line_width=1, color="red", legend_label="fitting lines", line_alpha=0.5)
    
    # plot the optimal crossover points
    x_cr = O[0, :]
    y_cr = y[J[T[0], 1:-1]]
    p.scatter(x_cr, y_cr, size=12, color="orange", alpha=0.8, marker="circle", legend_label="Optimal crossover")
    
    # set the legend position
    p.legend.location = "top_right"
    p.legend.title = "Legend"

    # edit the grids
    p.xgrid.grid_line_alpha=0.3
    p.ygrid.grid_line_alpha=0.3
    p.xgrid.grid_line_dash=[5, 5]
    p.ygrid.grid_line_dash=[5, 5]

    # return all
    return O, p



def butt_filt(y, cutoffs, sampling_frequency, order=4, type='lowpass', phase_corrected=True, plot=False):
    """
    Provides a convenient function to call a Butterworth filter with the specified parameters.

    Input:
        y:                  (1D numpy array)
                            the signal to be filtered.
        
        cutoffs:            (float, list, numpy 1D array)
                            the filter cutoff(s) in Hz.
        
        sampling_frequency: (float)
                            the sampling frequency of the signal in Hz.
        
        type:               (str)
                            a string defining the type of the filter: e.g. "low", "high", "bandpass", etc.
        
        phase_corrected:    (bool)
                            should the filter be applied twice in opposite directions to correct for phase lag?
        
        plot:               (bool)
                            Should a Bokeh Figure be returned representing the effect of the filter?

    Output:
        yf:                 (1D numpy array)
                            the resulting 1D filtered signal

        p:                  (Bokeh figure, optional)
                            a figure reflecting the effect of the filter
    """

    # get the filter coefficients
    sos = ss.butter(order, (np.array([cutoffs]).flatten() / (0.5 * sampling_frequency)), type, output="sos")

    # get the filtered data
    if phase_corrected:
        yf = ss.sosfiltfilt(sos, y)
    else:
        yf = ss.sosfilt(sos, y)
    
        # return the crossovers
    if not plot:
        return yf
    
    # generate the time-domain figure
    t = figure(width=fig_size, height=fig_size, title="Signal domain")
                
    # edit the axes labels
    t.xaxis.axis_label = "Samples (#)"
    t.yaxis.axis_label = "Y"

    # plot the raw data (time domain)
    x = np.arange(len(y))
    t.scatter(x, y, size=4, color="navy", alpha=0.5, marker="circle", legend_label="raw signal")
    t.line(x, y, line_width=2, color="navy", alpha=0.5, line_dash=(3, 3), legend_label="raw signal")
    
    # plot the filtered data (time domain)
    t.scatter(x, yf, size=4, color="red", alpha=0.5, marker="circle", legend_label="filtered signal")
    t.line(x, yf, line_width=2, color="red", alpha=0.5, line_dash=(3, 3), legend_label="filtered signal")

    # set the legend position
    t.legend.location = "top_right"
    t.legend.title = "Legend"

    # edit the grids
    t.xgrid.grid_line_alpha=0.3
    t.ygrid.grid_line_alpha=0.3
    t.xgrid.grid_line_dash=[5, 5]
    t.ygrid.grid_line_dash=[5, 5]

    # generate the frequency-domain figure
    s= figure(width=fig_size, height=fig_size, title="Frequency domain")
    
    # edit the axes labels
    s.xaxis.axis_label = "Frequency (Hz)"
    s.yaxis.axis_label = "Power Spectral Density"

    # plot the raw data (frequency domain)
    p, f = psd(y, sampling_frequency=sampling_frequency, plot=False)
    s.scatter(f, p, size=4, color="navy", alpha=0.5, marker="circle", legend_label="raw signal")
    s.line(f, p, line_width=2, color="navy", alpha=0.5, line_dash=(3, 3), legend_label="raw signal")
    
    # plot the filtered data (frequency domain)
    pf, ff = psd(yf, sampling_frequency=sampling_frequency, plot=False)
    s.scatter(ff, pf, size=4, color="red", alpha=0.5, marker="circle", legend_label="filtered signal")
    s.line(ff, pf, line_width=2, color="red", alpha=0.5, line_dash=(3, 3), legend_label="filtered signal")

    # generate a Label containing all the filter specs
    lines = ['{:25}'.format("Family: Butterworth"),
             '{:25}'.format("Type: " + type),
             '{:25}'.format("Order: {:0.0f}".format(order) + ((" (Phase-corrected) ") if phase_corrected else "")),
             '{:25}'.format("".join(["Cut-off: ", "(" if len(np.array([cutoffs]).flatten()) > 1 else "",
                                     ", ".join(["{:0.3f}".format(i) for i in np.array([cutoffs]).flatten()]),
                                     ")" if len(np.array([cutoffs]).flatten()) > 1 else "", " Hz"]))]   
    xs = np.tile((np.max(ff) - np.min(ff)) * 0.2 + np.min(ff), len(lines))
    ys = np.max(pf) - (np.max(pf) - np.min(pf)) * np.linspace(0.05, 0.25, len(lines))
    src = ColumnDataSource(data={'x': xs, 'y': ys, 't': lines})
    s.add_layout(LabelSet(x='x', y='y', text='t', text_align="left", text_font_size='10pt', text_alpha=0.5,
                          source=src))

    # hide the legend
    s.legend.visible = False

    # edit the grids
    s.xgrid.grid_line_alpha=0.3
    s.ygrid.grid_line_alpha=0.3
    s.xgrid.grid_line_dash=[5, 5]
    s.ygrid.grid_line_dash=[5, 5]

    # pack the 2 figures
    l = gridplot([t, s], ncols=1, toolbar_location="right", merge_tools=True)

    # return all
    return yf, l



def psd(y, sampling_frequency=1, n=None, plot=False):
    """
    compute the power spectrum of y using fft

    Input:
        y:      (ndarray)
                A 1D numpy array

        fs:     (float)
                the sampling frequency

        n:      (None, int)
                the number of samples to be used for FFT. if None, the length of y is used.

        plot:   (bool)
                should the output be returned into a figure?

    Output:
        ps:    (ndarray)
                the power of each frequency

        frq:    (ndarray)
                the frequencies.
    """

    # set n
    if n is None: n = len(y)

    # get the FFT and normalize by the length of y
    f = np.fft.rfft(y - np.mean(y), n) / len(y)

    # get the amplitude of the signal
    a = abs(f)

    # get the power of the signal
    ps = np.concatenate([[a[0]], 2 * a[1:-1], [a[-1]]]).flatten() ** 2

    # get the frequencies
    frq = np.linspace(0, sampling_frequency / 2, len(ps))

    # return the data
    if not plot:
        return ps, frq

    # generate the signal-domain figure
    q = figure(width=fig_size, height=fig_size, title="Signal")
                
    # edit the axes labels
    q.xaxis.axis_label = "Samples (#)"
    q.yaxis.axis_label = "Signal amplitude"

    # plot the psd
    x = np.arange(len(y))
    q.scatter(x, y, size=4, color="navy", alpha=0.5, marker="circle")
    q.line(x, y, line_width=2, color="navy", alpha=0.5, line_dash=(3, 3))
    
    # edit the grids
    q.xgrid.grid_line_alpha=0.3
    q.ygrid.grid_line_alpha=0.3
    q.xgrid.grid_line_dash=[5, 5]
    q.ygrid.grid_line_dash=[5, 5]

    # generate the frequency-domain figure
    p = figure(width=fig_size, height=fig_size, title="Power Spectral Density")
                
    # edit the axes labels
    p.xaxis.axis_label = "Sampling frequency units"
    p.yaxis.axis_label = "Power Spectral Density"

    # plot the psd
    p.scatter(frq, ps, size=4, color="navy", alpha=0.5, marker="circle")
    p.line(frq, ps, line_width=2, color="navy", alpha=0.5, line_dash=(3, 3))
    
    # edit the grids
    p.xgrid.grid_line_alpha=0.3
    p.ygrid.grid_line_alpha=0.3
    p.xgrid.grid_line_dash=[5, 5]
    p.ygrid.grid_line_dash=[5, 5]

    # merge the figures
    z = gridplot([q, p], ncols=1, toolbar_location="right", merge_tools=True)

    # return all
    return ps, frq, z



def find_peaks(y, height=None, plot=False):
    """
    detect the location (in sample units) of the peaks within y.
        
    Input:
        y:      (1D array)
                a 1D signal
            
        height: (float, None)
                a scalar defining the minimum height to be considered for a valid peak.
                If None, all peaks are returned.

        plot:   (bool)
                should the outcomes be returned as Bokeh figure?
                
    """

    # ensure y is a flat 1D array
    yf = np.array([y]).flatten()

    # set height
    if height is None:
        h = np.min(yf)
    else:
        h = height

    # get the first derivative of the signal
    d1 = yf[1:] - yf[:-1]
       
    # get the sign of the first derivative
    d1[d1 == 0] = 1
    sn = d1 / abs(d1)

    # get the peaks
    zc = np.argwhere(sn[1:] - sn[:-1] == -2).flatten()

    # exclude all peaks below the set height
    pks = zc[yf[zc] >= h] + 1
        
    # return the peaks
    if not plot:    
        return pks
    
    # generate the signal-domain figure
    p = figure(width=fig_size, height=fig_size, title="Peaks", toolbar_location="right")
                
    # edit the axes labels
    p.xaxis.axis_label = "Samples (#)"
    p.yaxis.axis_label = "Signal amplitude"

    # plot the signal
    x = np.arange(len(y))
    p.scatter(x, y, size=4, color="navy", alpha=0.5, marker="circle")
    p.line(x, y, line_width=2, color="navy", alpha=0.5, line_dash=(3, 3))
    
    # plot the peaks
    p.scatter(pks, y[pks], size=12, color="red", alpha=0.5, marker="cross")

    # edit the grids
    p.xgrid.grid_line_alpha=0.3
    p.ygrid.grid_line_alpha=0.3
    p.xgrid.grid_line_dash=[5, 5]
    p.ygrid.grid_line_dash=[5, 5]
       
    # return all
    return pks, p



def nextpow(n, b):
    """
    calculate the power at which the base has to be elevated in order to get the next value to n.

    Input:
        n: (float)
            a number
        b: (float)
            the base of the power.

    Output:
        k: (float)
            the value verifying the equation: n = b ** k --> k = log(n) / log(b)
    """
    return np.log(n) / np.log(b)



def crossings(y, value=0., x=None, interpolate=False, plot=False):
    """
    Dectect the crossing points in x compared to value.

    Input:
        y:              (1D array)
                        the data.

        value:          (float or 1D array with the same shape of y)
                        the value/s to be used to detect the crossings. If value is an array, the function
                        will find those points in x crossing values according to the value of values at a
                        given location.

        x:              (None or 1D array)
                        the x-value to be used. If None, an array of int is used to reflect the x-coordinate
                        of y at each sample.
        
        interpolate:    (bool)
                        Should the crossings location be interpolated? if False, no interpolation is performed
                        and the resulting output will correspond to the location of the samples immediately
                        after the crossings. If True, linear interpolation is performed to provide more accurate
                        crossing coordinates. 

        plot:           (bool)
                        if True a figure is returned in addition to the crossings reflecting the output of this
                        method.

    Output:
        a numpy array with the location of the crossing points
    """

    # get the sign of the signal without the offset
    sn = np.copy(y - value)
    sn[sn == 0.] = 1
    sn = sn / abs(sn)
    
    # get the location of the crossings
    cr = np.argwhere(abs(sn[1:] - sn[:-1]) == 2).flatten()
    
    # get the x coordinates
    if x is None:
        x = np.arange(len(y))

    # interpolate
    if interpolate:
        xcr = []
        for i, v in cr:
            
            # get the linear fitting coefficients: y = ax + b
            a = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
            b = y[i] - a * x[i]

            # obtain the x coordinate resulting in y = 0: --> x = -b / a
            xcr += [-b / a]

            # make the array
        xcr = np.array(xcr)

    # no interpolation is required
    else:
        xcr = x[cr]

    # return the crossings
    if not plot:
        return xcr
    
    # generate the signal-domain figure
    p = figure(width=fig_size, height=fig_size, title="Crossings", toolbar_location="right")
                
    # edit the axes labels
    p.xaxis.axis_label = "Samples (#)"
    p.yaxis.axis_label = "Signal amplitude"

    # plot the value
    p.line(x, np.tile(value, len(x)), line_width=1, color="black", alpha=0.3, line_dash=(2, 5))

    # plot the signal
    p.scatter(x, y, size=4, color="navy", alpha=0.5, marker="circle", legend_label="Signal")
    p.line(x, y, line_width=2, color="navy", alpha=0.5, line_dash=(3, 3), legend_label="Signal")
    
    # plot the crossings
    if interpolate:
        ycr = np.tile(value, len(xcr))
    else:
        ycr = y[cr]
    p.scatter(xcr, ycr, size=12, color="red", alpha=0.5, marker="cross", legend_label="Crossings")

    # edit the grids
    p.xgrid.grid_line_alpha=0.3
    p.ygrid.grid_line_alpha=0.3
    p.xgrid.grid_line_dash=[5, 5]
    p.ygrid.grid_line_dash=[5, 5]
       
    # return all
    return xcr, p


'''
def xcorr(X, c_type='unbiased', return_negative=False):
    """
    set the cross correlation of the data in X

    Input:
                     X : (P x N numpy array)
                         P = the number of variables
                         N = the number of samples

                c_type : str
                         {'biased', 'unbiased'}

        return_negative: bool
                         Should the negative lags be reported?

    Note:
        if X is a 1 x N array or an N length 1-D array, the autocorrelation is
        provided.
    """
    import numpy as np
    from scipy.signal import fftconvolve as fftxcorr

    # ensure the shape of X
    X = np.atleast_2d(X)

    # take the autocorrelation if X is a 1-d signal
    if X.shape[0] == 1:
        X = np.vstack((X, X))
    P, N = X.shape

    # remove the mean from each dimension
    V = X - np.atleast_2d(np.mean(X, 1)).T

    # take the cross correlation
    xc = []
    for i in np.arange(P - 1):
        for j in np.arange(i + 1, P):

            # FFT convolution
            R = fftxcorr(V[i], V[j][::-1], "full")

            # store the value
            R = np.atleast_2d(R)
            xc = np.vstack((xc, R)) if len(xc) > 0 else np.copy(R)

    # average over all the multiples
    xc = np.mean(xc, 0)

    # adjust the output
    lags = np.arange(-(N - 1), N)
    if not return_negative:
        xc = xc[(N - 1):]
        lags = lags[(N - 1):]

    # normalize
    if c_type == 'unbiased':
        xc /= (N + 1 - abs(lags))
    elif c_type == 'biased':
        xc /= (N + 1)
    else:
        st = 'The "c_type" parameter was not correctly specified. The "biased"'
        st += ' estimator has been used'
        print(st)
        xc /= (N + 1)
    return xc, lags

'''