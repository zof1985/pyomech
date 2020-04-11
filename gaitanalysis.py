


# IMPORTS


import os
import numpy as np
import pandas as pd
import pyomech.processing as pr
import pyomech.vectors as pv
import pyomech.utils as pu
import colorcet as cc
from bokeh.plotting import *
from bokeh.layouts import *
from bokeh.models import *
from bokeh.palettes import *
from bokeh.io import *



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
        return "\n".join(["{:25s}{:0.3f}".format(i + ":", getattr(self, i)) for i in Step.attr_names()])



    # MATH OPERATORS



    def __sub__(self, *args):
        if self.isvalid() and args[0].isvalid():
            return Step(**{i: getattr(self, i) - getattr(args[0], i) for i in Step.events_names()})
        else:
            return Step()



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



    def __init__(self, **kwargs):

        # initialize the errors dict
        self.errors = {i: np.array([]) for i in Step.events_names()[:-1]}

        # treadmill data was provided and the "new algorithm" is required
        if "speed" in np.array([i for i in kwargs.keys()]):
            self.__from_treadmill__(**kwargs)
            self.source = "New"
        
        # force data was provided
        elif "force" in np.array([i for i in kwargs.keys()]):
            self.__from_lab__(**kwargs)
            self.source = "Lab"

        # blackbox data was provided
        elif "blackbox" in np.array([i for i in kwargs.keys()]):
            self.__from_blackbox__(**kwargs)
            self.source = "Old"
        
        # the provided parameters are unknown thus create an empty object
        else:
            self.steps = []
            self.source = ""
            self.tw = None
            self.n = None
            self.fc = None  



    def __from_lab__(self, force, heel_right=None, meta_right=None, toe_right=None, heel_left=None, meta_left=None,
                     toe_left=None, tw=None, fc=10, n=4):
        """
        method extracting the gait events from kinematic data of the feet and the resultant force coming from a
        force platform.

        Input:
            force:              (Vector)
                                the vector containing the resultant force to be used for the analysis.
            
            heel_right/left:    (Vector)
                                the vector containing the 3D position in space of the heel. If not None, this is used
                                to estimate the foot-strike.
            
            meta_right/left:    (Vector)
                                the vector containing the 3D position in space of the 5th metatarsal head. If not None,
                                this is used to estimate the foot-strike.

            toe_right/left:     (Vector)
                                the vector containing the 3D position in space of longest toe of the foot. If not None,
                                this is used to estimate the toe-off.

            tw:                 (float)
                                not used. Parameter left only for compatibility with the other methods.

            fc:                 (float)
                                the cut-off frequency used to low-pass filter the data via a phase-corrected
                                Butterworth filter.

            n:                  (int)
                                the order of the phase-corrected Butterworth low-pass filter.
        """
        

        # store the additional parameters
        self.tw = tw
        self.fc = fc
        self.n = n
        
        """

        ########    FOOT-STRIKES    ########

        
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
                pks_Z = pr.find_peaks(self.__scale__(vct['Z'].values.flatten()), height=0.5, plot=False)

                # obtain the minima in the vertical direction
                mns_Y = pr.find_peaks(-self.__scale__(vct['Y'].values.flatten()), height=-0.1, plot=False)

                # for each peak in pks_Z find the closest minima in mns_Y.
                return [mns_Y[np.argmin(abs(mns_Y - i))] for i in pks_Z]


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
        fs_right = get_foot_strikes(heel_right, meta_right)
        fs_left = get_foot_strikes(heel_left, meta_left)
        
        # keep unique events
        fs_x = np.sort(np.unique(np.append(fs_right, fs_left)))
        fs = heel_right.index.to_numpy()[fs_x]
                                
        
        ########    MID-STANCE    ########


        # get the filtered force
        ff = force.butt_filt(cutoffs=self.fc, order=self.n, type="lowpass", phase_corrected=True, plot=False)

        # get the first minima in the anterior-posterior direction after fs_x 
        mns_fz = pr.find_peaks(-self.__scale__(force['Z'].values.flatten()), height=-0.5, plot=False)
        mns_fz = [[i for i in mns_fz if i > j][0] for j in fs_x[:-1]]

        # get the first zero after each mns_zf
        zrs_fz = pr.crossings(force['Z'].values.flatten(), plot=False)
        zrs_fz = [[i for i in zrs_fz if i > j][0] for j in mns_fz]

        # get the closest peak in Fy to each zrs_fz
        pks_fy = pr.find_peaks(self.__scale__(force['Y'].values.flatten()), height=0.5, plot=False)
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
            mns_z = pr.find_peaks(-self.__scale__(toe['Z'].values.flatten()), plot=False)

            # obtain the minima in the vertical direction
            mns_y = pr.find_peaks(-self.__scale__(toe['Y'].values.flatten()), height=-0.1, plot=False)
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
        """

        # get force first derivative
        d1 = force.der1_winter()
        
        # get the peaks in the first derivative and in force with height above 0.66
        pks_dv = pr.find_peaks(self.__scale__(d1['Y'].values.flatten()), height=0.66, plot=False)

        # get the force vertical crossings at the 5% of the force peak
        crs_fv = pr.crossings(force['Y'].values.flatten(), np.max(force['Y'].values.flatten()) * 0.05, plot=False)

        # get the closer crossing point to each peak
        fs_x = np.unique([crs_fv[np.argmin(abs(crs_fv - i))] for i in pks_dv])
        fs = force.index.to_numpy()[fs_x]

        # get the first point after a foot-strike in crs_fv
        to_x = np.unique([crs_fv[crs_fv > i][0] for i in fs_x[:-1]])
        to = force.index.to_numpy()[to_x]
        
        # get the peaks of the anterior-posterior force
        pks_fz = pr.find_peaks(force['Z'].values.flatten(), plot=False)

        # search for the minima in the anterior posterior direction between each fs and to, then get the first peak
        # after that minima
        ms_x = []
        for i, v in zip(fs_x[:-1], to_x):
            mn = np.argmin(force['Z'].values.flatten()[np.arange(i, v)]) + i
            ms_x += [[i for i in pks_fz if i > mn][0]]
        ms = force.index.to_numpy()[ms_x]

                
        ########    STORE THE STEPS    ########


        self.__add_steps__(fs, ms, to)
        


    def __from_blackbox__(self, blackbox, tw=None, fc=None, n=None):
        """
        method extracting the gait events from blackbox data.

        Input:
            blackbox:   (Vector)
                        the vector containing the blackbox data to be used for the analysis.

                        
            tw:     (float)
                    not used. Parameter left only for compatibility with the other methods.


            fc:     (float)
                    not used. Parameter left only for compatibility with the other methods.

            n:      (int)
                    not used. Parameter left only for compatibility with the other methods.
        """

        # add the parameters
        self.tw = tw
        self.fc = fc
        self.n = n

        # get the foot-strikes
        fs_loc = np.append([0], np.diff(blackbox['stato_w'].values.flatten())) == 1
        fs = blackbox.loc[fs_loc].index.to_numpy() + self.blackbox_delays['stato_w']
            
        # get the mid-stances
        ms_loc = np.append([0], np.diff(blackbox['wmin'].values.flatten())) == 1
        ms = blackbox.loc[ms_loc].index.to_numpy() + self.blackbox_delays['wmin']
        
        # get the toe-offs
        to_loc = np.append([0], np.diff(blackbox['wmax'].values.flatten())) == 1
        to = blackbox.loc[to_loc].index.to_numpy() + self.blackbox_delays['wmin']
        
        # store the steps and the errors
        self.__add_steps__(fs, ms, to)

        
        
    def __from_treadmill__(self, speed, tw=2, fc=10, n=2):
        """
        method extracting the gait events from blackbox data.

        Input:
            speed:  (Vector)
                    the vector containing the treadmill speed data to be used for the analysis.

            tw:     (float)
                    the time-window (in seconds) used to search for the next step

            fc:     (float)
                    the cut-off frequency used to low-pass filter the data via a phase-corrected Butterworth filter.

            n:      (int)
                    the order of the phase-corrected Butterworth low-pass filter.
        """


        def get_fs(sp):
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
            fc = self.fc
            while True:

                # get the filtered signal
                spf = pr.butt_filt(sp, cutoffs=fc, order=self.n,
                                   sampling_frequency=speed.sampling_frequency, plot=False)

                # get the first minima in spf lower than 0.4
                mn = pr.find_peaks(-self.__scale__(spf), height=-0.4, plot=False)[0]

                # get the last peak in the first derivative of spf before mn
                pks = pr.find_peaks((spf[2:] - spf[:-2])[:mn], plot=False)

                # check if pks is not empty
                if len(pks) > 0:

                    # return the first peak
                    return pks[-1] + 1
                
                # increase the cutoff frequency and repeat
                else:
                    fc += 1


        def get_ms(spf):
            """
            get the minima in the speed signal 
            
            Input:
                spf:    (1D array)
                        the speed signal

            Output:
                mn:     (int)
                        the location of the mid-stance in sample points
            """
            
            # get the first minima lower than 0.4
            mn =  pr.find_peaks(-spf, -0.4, plot=False)[0]
            return mn
            
            # get the first peak in the first derivative after mn
            # pk = pr.find_peaks((spf[2:] - spf[:-2])[mn:], plot=False)[0] + mn + 1

            # return the average of both
            # return int(np.round(np.mean([mn, pk])))
        
        
        def get_to(spf):
            """
            get the first peak in the filtered signal having amplitude above the 80% of the overall signal amplitude 
            
            Input:
                spf:    (1D array)
                        the (filtered) speed signal

            Output:
                pk:     (int)
                        the location of the toe-off in sample points
            """

            # get the first peak higher than 0.8 in the filtered speed
            pk = pr.find_peaks(spf, 0.8, plot=False)[0]
            return pk

            # get the first derivative
            # d1 = spf[2:] - spf[:-2]

            # get the first minima in the first derivative after pk
            # mn = pr.find_peaks(-d1[pk:], plot=False)[0] + 1 + pk

            # return int(np.round(np.mean([pk, mn])))


        # store the data
        self.tw = tw
        self.n = n
        self.fc = fc

        # initialize the steps output
        self.steps = []
        last = -1

        # start the simulation
        time = speed.index.to_numpy()
        ix_buf = np.argwhere((time >= 0) & (time <= self.tw)).flatten().tolist()
        while last is not None and last < time[-1]:

            # update the buffer index
            ix_buf = np.unique(np.append(ix_buf[1:], [np.min([ix_buf[-1] + 1, len(time) - 1])]))

            # check the gaitcycles list to see if a new search for new gait events should run
            # since the speed data are filtered. A minimum length is required.
            if len(ix_buf) <= 11:
                last = None
            elif time[ix_buf[0]] >= last:

                # get the data in the buffer and filter the speed data
                tm_buf = time[ix_buf]

                # get the scaled speed signal
                sp_buf = self.__scale__(speed.values.flatten()[ix_buf])

                # get the scaled and filtered speed signal
                spf_buf = self.__scale__(pr.butt_filt(speed.values.flatten()[ix_buf], self.fc, 
                                                      speed.sampling_frequency, self.n, plot=False))
                                                
                # find the foot-strike
                if last < 0:

                    # ensure the signal starts from a toe-off
                    t0 = get_to(spf_buf)
                    tm_buf = tm_buf[t0:]
                    sp_buf = sp_buf[t0:]
                    spf_buf = spf_buf[t0:]

                    # get the foot-strike
                    try:
                        fs0 = tm_buf[get_fs(sp_buf)]
                    except Exception:
                        fs0 = None
                else:
                    fs0 = self.steps[-1].landing

                # find the mid-stance
                try:
                    ix = np.argwhere(tm_buf > fs0).flatten()
                    ms = tm_buf[ix][get_ms(spf_buf[ix])]
                except Exception:
                    ms = None
                
                # fing the toe-off
                try:
                    ix = np.argwhere(tm_buf > ms).flatten()
                    to = tm_buf[ix][get_to(spf_buf[ix])]
                except Exception:
                    to = None
                                
                # find the landing
                try:
                    ix = np.argwhere(tm_buf > to).flatten()
                    fs1 = tm_buf[ix][get_fs(sp_buf[ix])]
                except Exception:
                    fs1 = None
                
                # add the new step
                if fs0 is not None and ms is not None and to is not None and fs1 is not None:
                    self.steps += [Step(fs0, ms, to, fs1)]
                    last = self.steps[-1].landing
                else:
                    last = None
        


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
                
                errors:     (1D array)
                            the list of events in the provided list occuring before out

                clean:      (1D array)
                            the list of events remaining after out
            """

            after = events[events > reference]
            errors = events[events < reference]
            try:
                return after[0], errors, events[events > after[0]]
            except Exception:
                return -1, errors, []


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
            new_ms, err_ms, ms_buf = first_after(ms_buf, new_fs)
            self.errors['mid_stance'] = np.append(self.errors['mid_stance'], err_ms)
            if new_ms != -1:
                new_to, err_to, to_buf = first_after(to_buf, new_ms)
                self.errors['toe_off'] = np.append(self.errors['toe_off'], err_to)
                if new_to != -1:
                    new_ln, err_fs, fs_buf = first_after(fs_buf, new_to)
                    self.errors['foot_strike'] = np.append(self.errors['foot_strike'], err_fs)
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

        # create the errors dict
        self.errors = {'foot_strike': err_fs, 'mid_stance': err_ms, 'toe_off': err_to}

    

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
        return (x - np.min(x)) / (np.max(x) - np.min(x))



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
        new.errors = self.errors.copy()
        new.steps = self.steps
        new.tw = self.tw
        new.n = self.n
        new.fc = self.fc
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
        os.makedirs(pu.lvlup(file), exist_ok=True)

        # store the parameters
        params = pd.DataFrame({'source': self.source, 'tw': self.tw, 'n': self.n, 'fc': self.fc}, index=[0])
        pu.to_excel(file, params, '__params__')

        # ensure errors have the same length
        m = np.max([len(np.array([self.errors[i]]).flatten()) for i in self.errors])
        err_df = pd.DataFrame()
        for i in self.errors:
            v = np.zeros((m)).flatten()
            v[:len(np.array([self.errors[i]]).flatten())] = self.errors[i]
            err_df[i] = pd.Series(v)

        # store the errors
        pu.to_excel(file, err_df, '__errors__')
        
        # store the steps
        pu.to_excel(file, self.df(), '__steps__')

        

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
        dfs = pu.from_excel(file, sheets=['__params__', '__errors__', '__steps__'], **kwargs)

        # generate an empty RunningAnalysis object
        R = RunningAnalysis()

        # add the parameters
        [setattr(R, i, dfs['__params__'][i].values[0]) for i in dfs['__params__']]
        
        # add the errors
        for i in dfs['__errors__']:
            R.errors[i] = np.squeeze(dfs['__errors__'][i].values)

            # remove the zeros at the end
            loc = np.argwhere(R.errors[i] == 0).flatten()
            if len(loc) > 0:
                R.errors[i] = R.errors[i][:loc[-1]]

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



    @property
    def blackbox_delays(self):
        return {'stato_w': -0.002, 'wmin': -0.032, 'wmax': -0.032}