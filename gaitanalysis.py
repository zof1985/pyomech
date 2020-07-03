


# IMPORTS


import os
import numpy as np
import pandas as pd
import colorcet as cc
import scipy.signal as ss
import scipy.integrate as si
from bokeh.plotting import *
from bokeh.layouts import *
from bokeh.models import *
from bokeh.palettes import *
from bokeh.io import *
from .processing import *
from .utils import *



# CLASSES



class Step():



    foot_strike = None
    mid_stance = None
    toe_off = None
    landing = None
    side = None



    def __init__(self, foot_strike=None, mid_stance=None, toe_off=None, landing=None, side=None):

        # store the data
        self.foot_strike = foot_strike
        self.mid_stance = mid_stance
        self.toe_off = toe_off
        self.landing = landing
        self.side = side

    

    @property
    def contact_time(self):
        try:
            return self.toe_off - self.foot_strike
        except Exception:
            return None
    


    @property
    def propulsion_time(self):
        try:
            return self.toe_off - self.mid_stance
        except Exception:
            return None
    


    @property
    def flight_time(self):
        try:
            return self.landing - self.toe_off
        except Exception:
            return None
    


    @property
    def step_time(self):
        try:
            return self.landing - self.foot_strike
        except Exception:
            return None
    


    def copy(self):
        return Step(side=self.side, **{i: getattr(self, i) for i in Step.events_names()})



    def dict(self):
        return {i: getattr(self, i) for i in Step.attr_names()}



    def df(self):
        dt = self.dict()
        return pd.DataFrame({i: dt[i] for i in dt}, index=[0])



    @staticmethod
    def attr_names():
        return ['side'] + Step.events_names() + Step.biofeedback_names()
    


    @staticmethod
    def events_names():
        return ['foot_strike', 'mid_stance', 'toe_off', 'landing']



    @staticmethod
    def biofeedback_names():
        return ['contact_time', 'propulsion_time', 'flight_time', 'step_time']



    def isvalid(self):
        return np.all([getattr(self, e) is not None for e in Step.events_names()])
    


    def __repr__(self):
        return self.__str__()



    def __str__(self):
        txt = []
        for i in Step.attr_names():
            if getattr(self, i) is None:
                txt += ["{:25s}".format(i + ": ")]
            else:
                txt += ["{:25s}{:0.3f}".format(i + ":", getattr(self, i))]
        return "\n".join(txt)



    # MATH OPERATORS



    def __sub__(self, *args):
        classcheck(args[0], ['float', 'int', 'Step'])
        try:
            if self.isvalid() and args[0].isvalid():
                return Step(**{i: getattr(self, i) - getattr(args[0], i) for i in Step.events_names()})
            else:
                return Step()
        except Exception:
            return Step(**{i: getattr(self, i) - args[0] for i in Step.events_names()})



    def __add__(self, *args):
        classcheck(args[0], ['float', 'int', 'Step'])
        try:
            if self.isvalid() and args[0].isvalid():
                return Step(**{i: getattr(self, i) + getattr(args[0], i) for i in Step.events_names()})
            else:
                return Step()
        except Exception:
            return Step(**{i: getattr(self, i) + args[0] for i in Step.events_names()})



    def __mul__(self, *args):
        classcheck(args[0], ['float', 'int', 'Step'])
        try:
            if self.isvalid() and args[0].isvalid():
                return Step(**{i: getattr(self, i) * getattr(args[0], i) for i in Step.events_names()})
            else:
                return Step()
        except Exception:
            return Step(**{i: getattr(self, i) * args[0] for i in Step.events_names()})



    def __truediv__(self, *args):
        classcheck(args[0], ['float', 'int', 'Step'])
        try:
            if self.isvalid() and args[0].isvalid():
                return Step(**{i: getattr(self, i) / getattr(args[0], i) for i in Step.events_names()})
            else:
                return Step()
        except Exception:
            return Step(**{i: getattr(self, i) / args[0] for i in Step.events_names()})



    # LOGICAL OPERATORS



    def __lt__(self, *args, **kwargs):
        out = []
        for i in Step.events_names():
            try:
                out += [getattr(self, i) < getattr(args[0], i)]
            except Exception:
                out += [False]
        return np.all(out)



    def __le__(self, *args, **kwargs):
        out = []
        for i in Step.events_names():
            try:
                out += [getattr(self, i) <= getattr(args[0], i)]
            except Exception:
                out += [False]
        return np.all(out)



    def __eq__(self, *args, **kwargs):
        out = []
        for i in Step.events_names():
            try:
                out += [getattr(self, i) == getattr(args[0], i)]
            except Exception:
                out += [False]
        return np.all(out)



    def __ne__(self, *args, **kwargs):
        out = []
        for i in Step.events_names():
            try:
                out += [getattr(self, i) != getattr(args[0], i)]
            except Exception:
                out += [False]
        return np.all(out)



    def __ge__(self, *args, **kwargs):
        out = []
        for i in Step.events_names():
            try:
                out += [getattr(self, i) >= getattr(args[0], i)]
            except Exception:
                out += [False]
        return np.all(out)



    def __gt__(self, *args, **kwargs):
        out = []
        for i in Step.events_names():
            try:
                out += [getattr(self, i) > getattr(args[0], i)]
            except Exception:
                out += [False]
        return np.all(out)



class RunningAnalysis():



    def __init__(self, source="", **kwargs):
                
        # treadmill data was provided
        if np.any([i == "speed" for i in kwargs.keys()]):
            self.__from_treadmill__(**kwargs)
                    
        # force data was provided
        elif np.any([i == "force" for i in kwargs.keys()]):
            self.__from_force__(**kwargs)

        # blackbox data was provided
        elif np.any([i == "blackbox" for i in kwargs.keys()]):
            self.__from_blackbox__(**kwargs)
        
        # kinematic data was provided
        elif np.any([i == "heel_right" for i in kwargs.keys()]):
            self.__from_kinematics__(**kwargs)
        
        # the provided parameters are unknown thus create an empty object
        else:
            self.steps = []
            self.source = ""
            self.tw = None
            self.n = None
            self.fc = None
            self.fs = None
            self.th = None

        # add the new source
        self.source = source
        

    def __from_kinematics__(self, heel_right, toe_right, heel_left, toe_left, meta_right, meta_left):
        """
        method extracting the gait events from kinematic data.

        Input:
            heel_right/left:    (Vector)
                                the vector containing the 3D position in space of the heel.
            
            meta_right/left:    (Vector)
                                the vector containing the 3D position in space of the 5th metatarsal head. If not None,
                                this is used to estimate the foot-strike. If provided, the first relative minima
                                occurring between heels and metas is used to define the foot-strike. 

            toe_right/left:     (Vector)
                                the vector containing the 3D position in space of longest toe of the foot. If not None,
                                this is used to estimate the toe-off.
        """
        
        # store the additional parameters
        self.tw = None
        self.fc = None
        self.n = None
        self.th = None
        self.fs = heel_right.sampling_frequency


        # FOOT-STRIKES


        def get_fs(vct):
            '''
            extract the foot-strike from kinematic data (heel or meta).

            Input:
                vct:    (Vector)
                        a Vector object representing the position of the heel or of the 5th metatarsal head over
                        time.

            Output:
                fs:     (list)
                        the time instants representing the foot-strikes.
            '''

            # obtain the peaks in the anterior-posterior direction
            pks_Z = find_peaks(self.__scale__(vct['Z'].values.flatten()), height=0.5, plot=False)

            # obtain the minima in the vertical direction
            mns_Y = find_peaks(-self.__scale__(vct['Y'].values.flatten()), height=-0.1, plot=False)

            # for each peak in pks_Z find the closest minima in mns_Y.
            return np.unique([mns_Y[np.argmin(abs(mns_Y - i))] for i in pks_Z])      
        
        
        def get_foot_strikes(heel, meta):
            '''
            extract the foot-strikes from heel and 5th metatarsal heads over time and sort them properly.

            Input:
                heel:   (Vector)
                        the vector containing the 3D position in space of the heel.
            
                meta:   (Vector)
                        the vector containing the 3D position in space of the 5th metatarsal head.
            
            Output:
                fs:     (list)
                        the time instants of the foot-strikes.
            '''

            # get the estimates separately
            fs_heel = get_fs(heel)
            fs_meta = get_fs(meta)
            
            # for each pair, get the event occurring first in time
            fs = []
            ref = fs_heel if len(fs_heel) <= len(fs_meta) else fs_meta
            alt = fs_heel if len(fs_heel) > len(fs_meta) else fs_meta
            for i in ref:
                j = alt[np.argmin(abs(alt - i))]
                fs += [np.min([i, j])]

            # ensure to remove doubles
            return np.unique(fs)


        # get the foot-strikes estimates from heel and meta
        fs_right = get_foot_strikes(heel_right, meta_right) if meta_right is not None else get_fs(heel_right)
        fs_left = get_foot_strikes(heel_left, meta_left) if meta_left is not None else get_fs(heel_left)
        
        # keep unique events
        fs_x = np.sort(np.unique(np.append(fs_right, fs_left)))
        fs = heel_right.index.to_numpy()[fs_x]
                                
        
        # MID-STANCES


        # get the filtered force
        
        ms_x = np.unique([pks_fy[np.argmin(abs(pks_fy - i))] for i in zrs_fz])
        ms = ff.index.to_numpy()[ms_x]
        

        ########    TOE-OFFS    ########


        def get_toe_offs(toe):
            '''
            extract the toe-off from toe data.

            Input:
                toe:   (Vector)
                       the vector containing the 3D position in space of the toe.
                        
            Output:
                to:     (list)
                        the samples of the toe-offs.
            '''
            
            # obtain the minima in the anterior-posterior direction
            mns_z = find_peaks(-self.__scale__(toe['Z'].values.flatten()), plot=False)

            # obtain the minima in the vertical direction
            mns_y = find_peaks(-self.__scale__(toe['Y'].values.flatten()), height=-0.1, plot=False)
            '''
            import matplotlib.pyplot as pl
            pl.plot(toe.index.to_numpy(), self.__scale__(toe['Z'].values.flatten()))
            pl.plot(toe.index.to_numpy(), self.__scale__(toe['Y'].values.flatten()))
            pl.plot(toe.index.to_numpy()[mns_z], self.__scale__(toe['Z'].values.flatten())[mns_z], 'go', label='mns_z')
            pl.plot(toe.index.to_numpy()[mns_y], self.__scale__(toe['Y'].values.flatten())[mns_y], 'ro', label='mns_y')
            pl.legend()
            pl.show()
            '''
            # for each minima in mns_z find the closest minima in mns_y.
            return [mns_y[np.argmin(abs(mns_y - i))] for i in mns_z]


        # get the estimates separately
        to_right = get_toe_offs(toe_right)
        to_left = get_toe_offs(toe_left)
            
        # sort them and keep only unique evens
        to_x = np.sort(np.unique(np.append(to_right, to_left)))
        to = toe_right.index.to_numpy()[to_x]
                
        # store the steps
        self.__add_steps__(fs, ms, to)



    def __from_force__(self, force):
        """
        method extracting the gait events from force data coming from a force platform.

        Input:
            force:              (Vector)
                                the vector containing the resultant force to be used for the analysis.
        """
        

        # store the additional parameters
        self.tw = None
        self.fc = None
        self.n = None
        self.th = None
        self.fs = force.sampling_frequency

        # get force first derivative
        d1 = force.der1_winter()
        
        # get the peaks in the first derivative and in force with height above 0.66
        pks_dv = find_peaks(self.__scale__(d1['Y'].values.flatten()), height=0.66, plot=False)

        # get the force vertical crossings at the 5% of the force peak
        crs_fv = crossings(self.__scale__(force['Y'].values.flatten()), 0.15, plot=False)

        # get the closer crossing point to each peak
        fs_x = np.unique([crs_fv[np.argmin(abs(crs_fv - i))] for i in pks_dv])
        fs = force.index.to_numpy()[fs_x]

        # get the first point after a foot-strike in crs_fv
        to_x = np.unique([crs_fv[crs_fv > i][0] for i in fs_x[:-1]])
        to = force.index.to_numpy()[to_x]
        
        # get the peaks of the anterior-posterior force
        pks_fz = find_peaks(force['Z'].values.flatten(), plot=False)

        # search for the minima in the anterior posterior direction between each fs and to, then get the first peak
        # after that minima
        ms_x = []
        for i, v in zip(fs_x[:-1], to_x):
            mn = np.argmin(force['Z'].values.flatten()[np.arange(i, v)]) + i
            ms_x += [[i for i in pks_fz if i > mn][0]]
        ms = force.index.to_numpy()[ms_x]

                
        ########    STORE THE STEPS    ########


        self.__add_steps__(fs, ms, to)
        


    def __from_blackbox__(self, blackbox):
        """
        method extracting the gait events from blackbox data.

        Input:
            blackbox:   (Vector)
                        the vector containing the blackbox data to be used for the analysis.
        """

        # add the parameters
        self.tw = None
        self.fc = None
        self.n = None
        self.th = None
        self.fs = None

        # get the foot-strikes
        fs_loc = np.append([0], np.diff(blackbox['stato_w'].values.flatten())) == 1
        fs = blackbox.loc[fs_loc].index.to_numpy() + self.blackbox_delays['stato_w']
            
        # get the mid-stances
        ms_loc = np.append([0], np.diff(blackbox['wmin'].values.flatten())) == 1
        ms = blackbox.loc[ms_loc].index.to_numpy() + self.blackbox_delays['wmin']
        
        # get the toe-offs
        to_loc = np.append([0], np.diff(blackbox['wmax'].values.flatten())) == 1
        to = blackbox.loc[to_loc].index.to_numpy() + self.blackbox_delays['wmin']
        
        # store the steps
        self.__add_steps__(fs, ms, to)
  

     
    def __from_treadmill__(self, speed, tw=2, fc=10, n=2, th=0.8):
        """
        method extracting the gait events from treadmill data using the new2 approach.

        Input:
            speed:  (Vector)
                    the vector containing the treadmill speed data to be used for the analysis.

            tw:     (float)
                    the time-window (in seconds) used to search for the next step

            fc:     (float)
                    the cut-off frequency used to low-pass filter the data via a phase-corrected Butterworth filter.

            n:      (int)
                    the order of the phase-corrected Butterworth low-pass filter.

            th:     (float)
                    a scalar representing the amplitude threshold for the detection of the toe-offs (it reflects a
                    percentage of the signal amplitude).
        """


        ########    GAIT EVENTS DETECTING METHODS    ########


        def __get_fs__(sp):
            """
            iteratively smooth the signal with higher filter cut-off and search for a peak in its first derivative
            until a peak is found before reaching a local minima in speed with amplitude lower than its half 
            
            Input:
                sp:     (1D array)
                        the speed signal

            Output:
                pk:     (int)
                        the location of the foot-strike in sample points
            """
            
            # first check that we have enough samples
            if len(sp) < 11:
                return np.nan

            # initialize the searching cut-off frequency
            fc = min(20, self.fc)

            # start the iterative search of the foot-strike
            while fc <= 20:

                # get the filtered signal
                spf = butt_filt(sp, cutoffs=self.fc, order=self.n, sampling_frequency=self.fs, plot=False)

                # try to get the next mid-stance
                ms = __get_ms__(spf)

                # return NaN if ms is not found
                if np.isnan(ms): return np.nan

                # find the peaks in the first derivative within the next mid-stance
                pks = find_peaks((spf[2:] - spf[:-2])[:(ms - 1)], plot=False)
                
                # return the last peak if pks is not empty
                if len(pks) > 0: return pks[-1] + 1
                
                # otherwise increase the cutoff frequency and repeat
                else: fc += 1

            # something went wrong. Thus return nan
            return np.nan


        def __get_ms__(sp):
            """
            get the minima in the speed signal 
            
            Input:
                sp:     (1D array)
                        the filtered speed signal

            Output:
                mn:     (int)
                        the location of the mid-stance in sample points
            """
            try:
                # find the first flexion point (either a local maxima or minima)
                pk = np.sort(np.append(find_peaks(sp, plot=False), find_peaks(-sp, plot=False)))[0]

                # get the first point in the signal below th and after pk
                st = np.argwhere(self.__scale__(sp)[pk:] < self.th).flatten()[0] + pk
            
            except Exception:
                return np.nan

            # get the next toe-off or the last point in the sp if to is not found
            end = __get_to__(sp[st:])
            end = (end + st) if not np.isnan(end) else (len(sp) - 1)

            # get the minimum value in sp within end
            mn = np.argmin(sp[:end])
            
            # if mn corresponds to end return nan (the identification of the minima in the signal is not reliable)
            # otherwise return mn
            return np.nan if mn == end else mn
        
        
        def __get_to__(sp):
            """
            get the first peak in the filtered signal having amplitude above the 80% of the overall signal amplitude 
            
            Input:
                sp:     (1D array)
                        the (filtered) speed signal

            Output:
                pk:     (int)
                        the location of the toe-off in sample points
            """
            
            # get the peaks with amplitude above th
            pks = find_peaks(self.__scale__(sp), self.th, plot=False)

            # return the first peak or NaN
            return pks[0] if len(pks) > 0 else np.nan


        ########    SETUP    ########


        # store the data
        self.tw = tw                        # time-window
        self.n = n                          # filter order
        self.fc = fc                        # filter cut-off frequency
        self.th = th                        # amplitude threshold (this is used to detect mid-stances and toe-offs)
        self.fs = speed.sampling_frequency  # the frequency of the signal

        # initialize the steps output
        self.steps = []

        # get the time signal
        time = speed.index.to_numpy()

        # get the starting buffer index
        ix_buf = np.argwhere((time >= 0) & (time <= self.tw)).flatten().tolist()
        
        
        ########    START THE SIMULATION    ########


        proceed = True
        while proceed:

            # update the buffer index
            ix_buf = np.unique(np.append(ix_buf[1:], [np.min([ix_buf[-1] + 1, len(time) - 1])]))

            # check if the algorithm must run
            if len(ix_buf) < 11:
                proceed = False
            if proceed and (len(self.steps) == 0 or time[ix_buf[0]] >= self.steps[-1].landing):

                # get the data in the buffer and filter the speed data
                tm = time[ix_buf]

                # get the speed signal within the current buffer
                sp = speed.values.flatten()[ix_buf]

                # to speed-up the search calculate also a scaled and filtered copy of the speed signal
                sf = butt_filt(sp, cutoffs=self.fc, order=self.n, sampling_frequency=self.fs, plot=False)
                
                # ensure the signal starts from a toe-off
                if len(self.steps) == 0:
                    t0 = __get_to__(sf)
                    tm = tm[t0:]
                    sp = sp[t0:]
                    sf = sf[t0:]

                    # get the first "foot-strike"
                    fs_ix = __get_fs__(sp)
                    fs = np.nan if np.isnan(fs_ix) else tm[fs_ix]
                    
                # get the foot-strike as the last landing
                else:
                    fs = self.steps[-1].landing
                    fs_ix = 0

                # find the mid-stance
                if not np.isnan(fs):
                    ms_ix = __get_ms__(sf[fs_ix:]) + fs_ix
                    ms = np.nan if np.isnan(ms_ix) else tm[ms_ix]
                else:
                    ms = np.nan
                
                # fing the toe-off
                if not np.isnan(ms):
                    to_ix = __get_to__(sf[ms_ix:]) + ms_ix
                    to = np.nan if np.isnan(to_ix) else tm[to_ix]
                else:
                    to = np.nan
                                
                # find the landing
                if not np.isnan(to):
                    ln_ix = __get_fs__(sp[to_ix:]) + to_ix
                    ln = np.nan if np.isnan(ln_ix) else tm[ln_ix]
                else:
                    ln = np.nan
                
                # add the new step
                if not np.any([np.isnan(i) for i in [fs, ms, to, ln]]):
                    self.steps += [Step(fs, ms, to, ln)]
                else:
                    proceed = False


    def __add_steps__(self, fs, ms ,to):
        """
        ensure the the detected sequence of foot-strikes, mid-stances and toe-offs occur correctly
        and starts and ends from a foot-strike. Then create the sequence of steps.

        Note:
            This method is used only internally by "__from_kinetics__" and "__from_blackbox__"

        Input:
            fs: (1D array)
                the location (in time or samples) of the foot-strikes
            
            ms: (1D array)
                the location (in time or samples) of the mid-stances

            to: (1D array)
                the location (in time or samples) of the toe-offs

        Output:
            self.steps: (dict)
                        a dict of Step object where the key reflect the Step occurrence.
        """


        def last_before(events, reference):
            """
            get the last event occurring before the reference

            Input:
                events:     (1D array)
                            the list of time events

                reference:  (float)
                            the time reference

            Output:
                before:     (float)
                            the time of the last event in events occurring before reference or -1 if that event is
                            not found.
            """
            before = events[events < reference]
            try:
                return before[-1]
            except Exception:
                return -1


        def first_after(events, reference):
            """
            get the first event occurring after the reference
            Input:
                events:     (1D array)
                            the list of time events

                reference:  (float)
                            the time reference

            Output:
                after:      (float)
                            the time of the first event in events occurring after reference or -1 if that event is
                            not found.

                clean:      (1D array)
                            the list of events remaining after out
            """

            after = events[events > reference]
            try:
                return after[0], events[events > after[0]]
            except Exception:
                return -1, []


        # prepare the data
        fs_out = []
        ms_out = []
        to_out = []
        fs_buf = np.copy(fs)
        ms_buf = np.copy(ms)
        to_buf = np.copy(to)
        ln_buf = np.copy(fs_buf[1:])
            
        # ensure to correctly align the first gait events
        ok = False
        while not ok:
            to1 = last_before(to_buf, ln_buf[0])
            if to1 == -1:
                ln_buf = ln_buf[1:]
            else:
                ms1 = last_before(ms_buf, to_buf[0])
                if ms1 == -1:
                    to_buf = to_buf[1:]
                else:
                    fs1 = last_before(fs_buf, ms_buf[0])
                    if fs1 == -1:
                        ms_buf = ms_buf[1:]
                    else:
                        new_fs = fs1
                        ok = True

        # remove unnecessary events
        fs_buf = np.array(fs_buf[fs_buf > new_fs])
        ms_buf = np.array(ms_buf[ms_buf > new_fs])
        to_buf = np.array(to_buf[to_buf > new_fs])

        # iterate the seach of new events
        while ok:
            new_ms, ms_buf = first_after(ms_buf, new_fs)
            if new_ms != -1:
                new_to, to_buf = first_after(to_buf, new_ms)
                if new_to != -1:
                    new_ln, fs_buf = first_after(fs_buf, new_to)
                    if new_ln != -1: 

                        # a full event was found. Therefore add it to the output data
                        fs_out += [new_fs]
                        ms_out += [new_ms]
                        to_out += [new_to]

                        # set the new foot-strike as the newly identified landing
                        new_fs = new_ln
                    else:
                        ok = False
                else:
                    ok = False
            else:
                ok = False

        # add the last foot-strike (i.e. landing)
        fs_out += [new_fs]

        # create the Steps dict
        self.steps = [Step(fs_out[i], ms_out[i], to_out[i], fs_out[i + 1]) for i in np.arange(len(ms_out))]



    def __scale__(self, x):
        """
        return a copy of x scaled to the 0-1 range.

        Note:
            This function is internally used by "__from_kinetics__" and "__from_treadmill__"

        Input:
            x:  (1D array)
                the raw signal

        Output:
            y:  (1D array)
                the x signal scaled to lie in the [0, 1] range.
        """
        if np.max(x) == np.min(x):
            return np.zeros(x.shape)
        return (x - np.min(x)) / (np.max(x) - np.min(x))



    def __repr__(self):
        return self.__str__()



    def __str__(self):
        txt = self.df().__str__()
        txt += "\n\n"
        txt += "{:25s}".format("Source:") + self.source + "\n"
        txt += "{:25s}".format("Time window (s):") + (str(self.tw) if self.tw is not None else "") + "\n"
        txt += "{:25s}".format("Filter order:") + (str(self.n) if self.n is not None else "") + "\n"
        txt += "{:25s}".format("Filter cut-off (Hz):") + (str(self.fc) if self.fc is not None else "") + "\n"
        txt += "{:25s}".format("Amplitude threshold:") + (str(self.th) if self.th is not None else "") + "\n"
        txt += "{:25s}".format("Sampling frequency (Hz):") + (str(self.fs) if self.fs is not None else "")
        return txt



    def correct(self, **kwargs):
        """
        apply a correction value to the specified event(s)

        Input:
            kwargs: a named event corresponding to any of the Step.events_names attribute containing a float value
                    to be added to all the steps.
        
        Output:
            R:  (RunningAnalysis)
                a new RunningAnalysis instance containing the data of self with the required correction.
        """

        # copy
        R = self.copy()

        # apply the correction to each Step
        for i in np.arange(len(R.steps)):
            for event in kwargs:
                if event in Step.events_names():
                    try:
                        setattr(R.steps[i], event, getattr(R.steps[i], event) + kwargs[event])
                    except Exception:
                        setattr(R.steps[i], event, None)
        
        # return R
        return R



    def copy(self):
        new = RunningAnalysis()
        new.source = self.source
        new.steps = self.steps
        new.tw = self.tw
        new.n = self.n
        new.fc = self.fc
        new.th = self.th
        new.fs = self.fs
        return new



    def df(self):
        """
        get the detected steps in a pandas.DataFrame format.
        """

        # manage the case RunningAnalysis has no steps
        if len(self.steps) == 0:
            df = Step().df()
            df.insert(0, 'N', None)
            df.insert(0, 'Source', self.source)
            return df

        # put together all the steps
        dfs = pd.DataFrame()
        for i, v in enumerate(self.steps):
            df = v.df()
            df.insert(0, 'N', i + 1)
            df.insert(0, 'Source', self.source)
            dfs = dfs.append(df, ignore_index=True, sort=False)
        return dfs



    def to_excel(self, file):
        """
        export the current RunningAnalysis object to an excel file.
        
        Input:
            file:   (str)
                    the file path where the data are stored.
        """

        # ensure the file folder exists
        os.makedirs(lvlup(file), exist_ok=True)

        # store the parameters
        params = pd.DataFrame({'source': self.source, 'tw': self.tw, 'n': self.n, 'fc': self.fc, 'fs': self.fs,
                               'th': self.th}, index=[0])
        to_excel(file, params, '__params__')
        
        # store the steps
        to_excel(file, self.df(), '__steps__')


        
    @staticmethod
    def from_excel(file, **kwargs):
        """
        import a new RunningAnalysis object from csv file.
        
        Input:
            file:   (str)
                    the file path where the data are stored.
            
            kwargs: (dict)
                    additional parameters passed directly to pandas.read_excel.

        Output:
            a new RunningAnalysis instance object.
        """

        # get the data
        dfs = from_excel(file, sheets=['__params__', '__steps__'], **kwargs)

        # generate an empty RunningAnalysis object
        R = RunningAnalysis()

        # add the parameters
        [setattr(R, i, dfs['__params__'][i].values[0]) for i in dfs['__params__']]
        
        # add the steps
        for i in np.arange(dfs['__steps__'].shape[0]):
            R.steps += [Step(*dfs['__steps__'][np.append(Step.events_names(), ['side'])].values[i])]

        # return R
        return R



    @staticmethod
    def align(*args, reference):
        """
        align args to reference

        Input:
            reference:      (RunningAnalysis)
                            the running analysis object to be used as reference

            args:           (RunningAnalysis)
                            a number of RunningAnalysis objects to be aligned to reference.

        Output:
            aligned_args:   (RunningAnalysis)
                            the RunningAnalysis objects provided in args aligned to reference.
        """

        # check if some exclusions can be made
        if len(reference.steps) == 0:
            return args
        
        # difference function
        delta = lambda a, b: abs(a.foot_strike - b.foot_strike) if a.isvalid() and b.isvalid() else np.nan

        # iterate the search on each arg
        new_args = []
        for i in np.arange(len(args)):

            # get the steps
            steps = args[i].steps

            # create a list of empty steps with length equal to reference
            s_list = [Step() for ref in reference.steps]

            # for each step search the one at shortest distance in reference and progressively fill the gaps in s_list
            for step in steps:
                diffs = [delta(step, j) for j in reference.steps]
                if not np.all(np.isnan(diffs)):
                    c = np.nanargmin(diffs)
                    s_list[c] = step

            # add the aligned steps to the new arg
            new_arg = args[i].copy()
            new_arg.steps = s_list
            new_args += [new_arg]

        # return the aligned args
        return new_args 



    @staticmethod
    def draw(signals, reference, comparisons):
        """
        create a bokeh figure representing the signals and the events defined by references.

        Input:
            signals:        (dict)
                            a dict of Vector objects containing the signals to be represented over time.

            reference:      (RunningAnalysis)
                            the Running analysis to be used as reference.

            comparisons:    (list)
                            a list of RunningAnalysis objects that are compared.

        Output:
            p:              (bokeh.Figure)
                            a bokeh figure representing the data
        """


        def update_datasets(attr, old, new):
            """
            Internal method used to populate and update the plot dynamically.
            It automatically takes the data available from withing the method environment and then updates the data
            sources that are used to populate the figure.
            """


            def update_sources(label, dt):
                """
                internal function used to ensure correct add or update of the available data in sources.

                Input:
                    label:  (str)
                            the key of the sources dict that have to be updated

                    dt:     (dict)
                            the dict containing the data to update
                """
                if not np.any([label  == i for i in sources.keys()]):
                    sources[label] = ColumnDataSource()
                sources[label].data = dt

                    
            # get the x and y axis minima and maxima
            x_rng = [2e16, -2e16]
            y_rng = [2e16, -2e16]

            # update the signals data
            for t in signals:
                for dim in signals[t].columns:
                    x = signals[t].index.to_numpy()
                    y = reference.__scale__(signals[t][dim].values.flatten()) * 100
                    
                    # add the data to the output source
                    update_sources(" - ".join([t, dim]), {'x': x, 'y': y})

                    # update the x and y scale ranges
                    x_rng = [np.min(np.append(x, [x_rng[0]])), np.max(np.append(x, [x_rng[1]]))]
                    y_rng = [np.min(np.append(y, [y_rng[0]])), np.max(np.append(y, [y_rng[1]]))]

            # update the slider
            x_sli.start = x_rng[0]
            x_sli.end = x_rng[-1] - 1
            x_sli.value = x_rng[0]
            x_sli.step = (x_sli.end - x_sli.start) / 100

            # update the horizontal lines separating the patches corresponding to each RunningAnalysis being compared
            low = np.min(y_rng) - 0.1 * (np.max(y_rng) - np.min(y_rng))
            up = low + (np.max(y_rng) - np.min(y_rng)) * 1.2
            rng = (up - low) / len(comparisons)
            hor_src = {'x': [], 'y': []}
            for i in np.arange(1, len(comparisons)):
                hor_src['x'] += [x_rng]
                hor_src['y'] += [np.tile(low + rng * i, 2)]
            update_sources("horizontal", hor_src)

            # adjust the reference events data
            ref_src = {'Foot-strike (' + reference.source + ')': {'x': [], 'y': []},
                       'Mid-stance (' + reference.source + ')': {'x': [], 'y': []},
                       'Toe-off (' + reference.source + ')': {'x': [], 'y': [],}}
            refs = {'Foot-strike (' + reference.source + ')': 'foot_strike',
                       'Mid-stance (' + reference.source + ')': 'mid_stance',
                       'Toe-off (' + reference.source + ')': 'toe_off'}
            for i in reference.steps:
                if i.isvalid():
                    for j in ref_src:
                        ref_src[j]['x'] += [np.tile(getattr(i, refs[j]), 2)]
                        ref_src[j]['y'] += [[low, up]]
            for i in ref_src:
                update_sources(i, ref_src[i])

            # update the biofeedback data source
            for i, v in enumerate(comparisons):
                bio_src = {'Loading response (' + v.source + ')': {'x': [], 'y': []},
                           'Propulsion phase (' + v.source + ')': {'x': [], 'y': []},
                           'Flying phase (' + v.source + ')': {'x': [], 'y': []}}
                refs = {'Loading response (' + v.source + ')': ['foot_strike', 'mid_stance'],
                        'Propulsion phase (' + v.source + ')': ['mid_stance', 'toe_off'],
                        'Flying phase (' + v.source + ')': ['toe_off', 'landing']}
                bot = low + rng * i
                top = low + rng * (i + 1)
                for j in v.steps:
                    if j.isvalid():
                        for k in refs:
                            loc = (getattr(j, refs[k][0]), getattr(j, refs[k][1]))
                            bio_src[k]['x'] += [np.hstack((loc, loc[::-1]))]
                            bio_src[k]['y'] += [np.hstack((np.tile(bot, 2), np.tile(top, 2)))]
                for k in bio_src:
                    update_sources(k, bio_src[k])


        # create the figure
        fig = figure(width=1200, height=600, title="Running Analysis", x_range=(0, 1.0), y_range=(0, 100),
                     tools="wheel_zoom, box_zoom, reset", toolbar_location="above")

        # setup a slider to adjust the data x range
        x_sli = Slider(start=0, end=1, value=1, step=1, title="Time (s)")

        # ensure all vectors have the same time unit
        assert len(np.unique([signals[i].time_unit for i in signals])) == 1, "Signals must have the same time unit."

        # edit the grids
        fig.xgrid.grid_line_alpha=0.3
        fig.ygrid.grid_line_alpha=0.3
        fig.xgrid.grid_line_dash=[5, 5]
        fig.ygrid.grid_line_dash=[5, 5]
        
        # set the axis labels
        fig.xaxis.axis_label = "Time (" + signals[[i for i in signals][0]].time_unit + ")"
        fig.yaxis.axis_label = "Normalized units"

        # create the data source
        sources = {}
        update_datasets(None, None, None)
        
        # setup a callback for the slider
        sli_cb = CustomJS(args={'x_range': fig.x_range},
                          code="""
                               x_range.start = cb_obj.value;
                               x_range.end = cb_obj.value + 1;
                               """)
        x_sli.js_on_change('value', sli_cb)

        # initialize the renderers dict to make the legend
        rndrs = {}

        # plot the signals
        sig_col = cc.glasbey
        for t in signals:
            for dim in signals[t].columns:
                label = " - ".join([t, dim])
                rndrs[label] = [fig.scatter(x='x', y='y', size=3, color=sig_col[0], alpha=0.5,
                                           marker="circle", source=sources[label]),
                                fig.line(x='x', y='y', line_width=2, color=sig_col[0], alpha=0.5,
                                         line_dash=(3, 3), source=sources[label])]
                sig_col = sig_col[1:]
        
        # plot the reference steps
        for i, v in zip(['Foot-strike (' + reference.source + ')', 'Mid-stance (' + reference.source + ')',
                         'Toe-off (' + reference.source + ')'], ['solid', 'dashed', 'dotted']):
            rndrs[i] = [fig.multi_line(xs='x', ys='y', line_width=3, color='black', alpha=0.8, line_dash=v,
                                       source=sources[i])]
        
        # plot the horizontal lines (they do not have to be rendered in the legend)
        fig.multi_line(xs='x', ys='y', line_width=1, color='grey', alpha=0.5, line_dash=(2, 4),
                       source=sources['horizontal'])

        # plot the biofeedback
        lo_map = [cc.CET_L5[int(i)] for i in np.linspace(130, 255, len(comparisons))]
        pr_map = [cc.CET_L6[int(i)] for i in np.linspace(130, 255, len(comparisons))]
        fl_map = [cc.bmw[int(i)] for i in np.linspace(130, 255, len(comparisons))]
        dt = [('Loading response (' + v.source + ')', lo_map[i]) for i, v in enumerate(comparisons)]
        dt += [('Propulsion phase (' + v.source + ')', pr_map[i]) for i, v in enumerate(comparisons)]
        dt += [('Flying phase (' + v.source + ')', fl_map[i]) for i, v in enumerate(comparisons)]
        for i in dt:
            rndrs[i[0]] = [fig.patches(xs='x', ys='y', fill_alpha=0.2, line_alpha=0.6, fill_color=i[1],
                                       line_color=i[1], source=sources[i[0]])]
        
        # set the legend
        legend = Legend(items=[LegendItem(label=i, renderers=rndrs[i]) for i in rndrs], location="center",
                        title="Legend")
        legend.title_text_font_size = "12pt"
        legend.title_text_font_style = "bold"
        legend.click_policy = "hide"
        fig.add_layout(legend, "right")
        
        # set the output layout
        layout = column(fig, x_sli)

        # return the figure
        return layout



    def blackbox_delays(self):
        return {'stato_w': -0.002, 'wmin': -0.032, 'wmax': -0.032}
    


    def treadmill_delays(self, speed_kmh):
        """
        return the correction coefficients for the treadmill_new2 algorithm.
        
        Input:
            speed_kmh:  (float)
                        the actual selected speed in km/h.
        
        Output:
            correction: (dict)
                        A dict containing the correction coefficients for the foot-strike,
                        mid-stance, toe-off and landings.
        """
        return {
            'foot_strike': 0.01835 + 0.00098 * speed_kmh,
            'mid_stance': 0.00206 * speed_kmh - 0.00932,
            'toe_off': 0.05917 - 0.00393 * speed_kmh,
            'landing': 0.01835 + 0.00098 * speed_kmh
            }