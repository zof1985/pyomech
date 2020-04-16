# generic imports

import numpy as np
import pyomech.utils as ut
import pyomech.vectors as dt
from itertools import combinations
from scipy.spatial.transform import Rotation
from pandas import DataFrame, MultiIndex, IndexSlice


# CONSTANTS
g = 9.80665  # acceleration of gravity in m/s^-2


"""
The following coefficients corresponds to those described by de Leva (1996a) in his Table 4 and are used to estimate
the mass, the Centre of Mass (CoM) position and the gyration radius around each rotation axis.

Reference:
    de Leva P. (1996a) Adjustments to Zatiorsky-Seluyanov's segment inertia parameters. Journal of Biomechanics,
        29(9):1223-30.
"""

segments = {

    # Head segment is calculated from the head vertex to the spinous process of the 7th cervical vertebrae.
    'head': {
        'male':{
            'mass': 0.0694,
            'CoM': 0.5002,
            'gyration_radius_ap': 0.303,
            'gyration_radius_ml': 0.315,
            'gyration_radius_vt': 0.261
            },
        'female':{
            'mass': 0.0668,
            'CoM': 0.4841,
            'gyration_radius_ap': 0.271,
            'gyration_radius_ml': 0.295,
            'gyration_radius_vt': 0.263
            }
        },

    # Trunk segment is calculated from the 7th cervical vertebrae and the mid-point between the hips joint center.
    'trunk': {
        'male':{
            'mass': 0.4346,
            'CoM': 0.5138,
            'gyration_radius_ap': 0.328,
            'gyration_radius_ml': 0.306,
            'gyration_radius_vt': 0.169
            },
        'female':{
            'mass': 42.57,
            'CoM': 0.4964,
            'gyration_radius_ap': 0.307,
            'gyration_radius_ml': 0.292,
            'gyration_radius_vt': 0.147
            }
        },
    
    # The upper-arm segment is calculated from the Shoulder joint center to the elbow joint center of the same side.
    'arm': {
        'male':{
            'mass': 0.0271,
            'CoM': 0.5772,
            'gyration_radius_ap': 0.285,
            'gyration_radius_ml': 0.269,
            'gyration_radius_vt': 0.158
            },
        'female':{
            'mass': 0.0255,
            'CoM': 0.5754,
            'gyration_radius_ap': 0.278,
            'gyration_radius_ml': 0.260,
            'gyration_radius_vt': 0.148
            }
        },
    
    # The forearm segment is calculated from the elbow joint center to the wrist joint center of the same side.
    'forearm': {
        'male':{
            'mass': 0.0162,
            'CoM': 0.4574,
            'gyration_radius_ap': 0.276,
            'gyration_radius_ml': 0.265,
            'gyration_radius_vt': 0.121
            },
        'female':{
            'mass': 0.0138,
            'CoM': 0.4559,
            'gyration_radius_ap': 0.261,
            'gyration_radius_ml': 0.257,
            'gyration_radius_vt': 0.094
            }
        },
    
    # The hand segment is calculated from the wrist joint center to the end of the 3rd metacarpale of the same side.
    'hand': {
        'male':{
            'mass': 0.061,
            'CoM': 0.7900,
            'gyration_radius_ap': 0.628,
            'gyration_radius_ml': 0.513,
            'gyration_radius_vt': 0.401
            },
        'female':{
            'mass': 0.056,
            'CoM': 0.7474,
            'gyration_radius_ap': 0.531,
            'gyration_radius_ml': 0.454,
            'gyration_radius_vt': 0.335
            }
        },
    
    # The thigh segment is calculated from the hip joint center to the knee joint center of the same side.
    'thigh': {
        'male':{
            'mass': 0.1416,
            'CoM': 0.4095,
            'gyration_radius_ap': 0.329,
            'gyration_radius_ml': 0.329,
            'gyration_radius_vt': 0.149
            },
        'female':{
            'mass': 0.1478,
            'CoM': 0.3612,
            'gyration_radius_ap': 0.369,
            'gyration_radius_ml': 0.364,
            'gyration_radius_vt': 0.162
            }
        },
    
    # The shank segment is calculated from the knee joint center to the lateral malleolus of the same side.
    'shank': {
        'male':{
            'mass': 0.0432,
            'CoM': 0.4459,
            'gyration_radius_ap': 0.255,
            'gyration_radius_ml': 0.249,
            'gyration_radius_vt': 0.103
            },
        'female':{
            'mass': 0.0481,
            'CoM': 0.4416,
            'gyration_radius_ap': 0.271,
            'gyration_radius_ml': 0.267,
            'gyration_radius_vt': 0.093
            }
        },
    
    # The foot segment is calculated from the heel to the tip of the longest toe of the foot.
    'foot': {
        'male':{
            'mass': 0.0137,
            'CoM': 0.4415,
            'gyration_radius_ap': 0.257,
            'gyration_radius_ml': 0.245,
            'gyration_radius_vt': 0.124
            },
        'female':{
            'mass': 0.0129,
            'CoM': 0.4014,
            'gyration_radius_ap': 0.299,
            'gyration_radius_ml': 0.279,
            'gyration_radius_vt': 0.139
            }
        }
    }



class RigidBody():
    """
    A class representing a rigid body with its properties and geometrial features.
    """

    # generated variables
    mass = []
    CoM = []
    inertia = []
    description = ""

    # constructor
    def __init__(self, mass, CoM, inertia, description):
        """
        Input:
            mass:       (float)
                        the mass of the rigid body.

            CoM:        (3D Vector)
                        the vector defining the coordinates of the centre of mass.

            inertia:    (3D Vector)
                        a 3x3 DataFrame defining the inertia tensor of the current rigid body.

            description:(str)
                        a string describing the rigid body.
        """
        
        # check the mass    
        ut.classcheck(mass, ['float', 'int'])
        
        # check the CoM
        ut.classcheck(CoM, ['Vector'])
        assert CoM.shape[1] == 3, "'CoM' must be a 3D vector."
        
        # check the inertia
        ut.classcheck(inertia, ['Vector'])
        assert inertia.shape[1] == 3, "'CoM' must be a 3D vector."
        
        # perform CoMparisons
        txt = "'CoM' and 'inertia' must have the same dimensions."
        assert np.all([i in CoM.columns.to_numpy() for i in inertia.columns.to_numpy()]), txt
        txt = "'CoM' and 'inertia' must have the same index."
        assert CoM.shape[0] == inertia.shape[0], txt
        assert np.all([i in CoM.index.to_numpy() for i in inertia.index.to_numpy()]), txt
                
        # store the data
        self.mass =  mass
        self.CoM = CoM
        self.inertia = inertia
        self.description = description


    # method to generate a copy of the current object
    def copy(self):
        """
        Create a copy of the current Rigid Body.
        """
        return RigidBody(self.mass, self.CoM, self.inertia, self.description)

    
    # method to combined segments
    def combine(self, description, *args):
        """
        Add multiple rigid bodies to the current one and return the CoMbination of all.

        Input:
            description: (str)
                How to describe the new RigidBody.

            args: (RigidBody)
                1 or more RigidBody objects.

        Output:
            C:  (RigidBody)
                A new RigidBody CoMbining all the entered objects with the current one.
        """

        # check entries
        for i in args:
            ut.classcheck(i, ["RigidBody", "deLeva_RigidBody"])
            txt = "RigidBody.CoM objects dimensions are not consistent."
            assert self.CoM.shape[1] == i.CoM.shape[1], txt
            assert np.all([j in self.CoM.columns.to_numpy() for j in i.CoM.columns.to_numpy()]), txt
            txt = "RigidBody.CoM objects indices are not consistent."
            assert self.CoM.shape[0] == i.CoM.shape[0], txt
            assert np.all([j in self.CoM.index.to_numpy() for j in i.CoM.index.to_numpy()]), txt
            txt = "RigidBody.inertia objects dimensions are not consistent."
            assert self.inertia.shape[1] == i.inertia.shape[1], txt
            assert np.all([j in self.inertia.columns.to_numpy() for j in i.inertia.columns.to_numpy()]), txt
            txt = "RigidBody.inertia objects indices are not consistent."
            assert self.inertia.shape[0] == i.inertia.shape[0], txt
            assert np.all([j in self.inertia.index.to_numpy() for j in i.inertia.index.to_numpy()]), txt

        # set the new description
        C = self.copy()
        C.description = description

        # combine rigid bodies masses
        C.mass += np.sum([i.mass for i in args])
        
        # combine the CoMs
        C.CoM = self.CoM * self.mass
        for i in args:
            C.CoM = C.CoM + i.CoM * i.mass
        C.CoM = C.CoM * (1. / C.mass)
        
        # combine the inertia moments
        C.inertia = self.inertia + ((self.CoM - C.CoM) ** 2 * self.mass).values
        for i in args:
            C.inertia = C.inertia + i.inertia + ((i.CoM - C.CoM) ** 2 * i.mass).values

        # return the new RigidBody
        return C



class deLeva_RigidBody(RigidBody):
    """
    A RigidBody superclass generated using the de Leva (1996) estimations.
    """


    # constructor
    def __init__(self, origin, end, body_weight, male, what):
        """
        Input:
            origin: (3D Vector)
                    the vector with the data defining the position of the origin of the body segment.

            end:    (3D Vector)
                    the vector with the data defining the position of the end of the body segment.

            body_weight: (float)
                    the weight of the participant in kg.

            male:   (bool)
                    True if the participant is a male, False otherwise.

            what:   (str)
                    any of ["Head", "Trunk", "Arm", "Forearm", "Thigh", "Shank", "Foot"].
        """
        
        # Check the entered parameters
        for i in [origin, end]:
            ut.classcheck(i, ['Vector'])
            assert i.shape[1] == 3, "'origin' and 'end' must be a 3D vector."
        assert np.all([i in origin.df.columns for i in end.df.columns]), "'origin' and 'end' must have the same ndim."
        same_index = np.sum(np.diff(origin.index.to_numpy() - end.index.to_numpy())) == 0
        assert same_index, "'origin' and 'end' must have same index."
        ut.classcheck(body_weight, ['float', 'int'])
        assert male or not male, "'male' must be a boolean."
        txt = "'what' must by any of the following string: " + str([i for i in segments.keys()])
        assert what.lower() in [i for i in segments.keys()], txt

        # get the specific parameters for the current segment according to de Leva (1996)
        length = (origin - end).module().values.flatten()
        gender = 'male' if male else 'female'
        mass =  body_weight * segments[what][gender]['mass']
        CoM = (end - origin) * segments[what][gender]['CoM'] + origin
        gyr = np.array([segments[what][gender]['gyration_radius_ml'],
                        segments[what][gender]['gyration_radius_ap'],
                        segments[what][gender]['gyration_radius_vt']])
        idx = CoM.index.to_numpy()
        col = CoM.columns.to_numpy()
        inertia = {i: v for i, v in zip(col, np.atleast_2d([i * gyr for i in length]).T ** 2 * mass)}
        inertia = dt.Vector(inertia, idx, CoM.xunit, r'$kg\cdot ' + CoM.dunit + '^2$', 'Moment of inertia') 
        
        # generate the RigidyBody object
        super().__init__(mass, CoM, inertia, what.lower())



"""
The following coefficients corresponds to those described by de Leva (1996b) in his Table 2 and are used to estimate
the longitudinal distances of different joint centes according to external markers.

Reference:
    de Leva P. (1996b) Joint center longitudinal positions CoMputed from a selected subset of Chandler's data.
        Journal of Biomechanics, 29(9):1231-33
"""

joints = {

    # The shoulder joint centre is calculated from the acromion to the radial tubercle of the same side.
    'Shoulder': 0.104,

    # The elbow joint centre is calculated from the acromion to the radial tubercle of the same side.
    'Elbow': 0.957,
    
    # The wrist joint centre is calculated from the Elbow joint centre to the approximated wrist joint centre.
    'Wrist': 0.993,

    # The hip joint centre is calculated from the ASIS to the tibial process.
    'Hip': 0.198,

    # The knee joint centre is calculated from the great throcanter to the tibial process.
    'Knee': 0.926,

    # The ankle joint centre is calculated from the knee joint centre to the lateral malleoulus.
    'Ankle': 1.016,
    }



class ReferenceFrame():
    """
    Generate a Reference Frame object.
    """



    def __init__(self, i=None, j=None, k=None):
        """
        Input:
            i, j, k: (Vector)
                     The vectors each defining the origin and the versors of the reference frame. If 2 versors are
                     provided, the third is obtained via cross-product of the other 2.
        """

        # all data must be vectors
        ut.classcheck(i, ['Vector', 'NoneType'])
        ut.classcheck(j, ['Vector', 'NoneType'])
        ut.classcheck(k, ['Vector', 'NoneType'])

        # check the arguments
        versors = [n for n in [i, j ,k] if n is not None]
        assert len(versors) >= 2, "A minimum of 2 versors must be provided."
        for n in versors[1:]:
            versors[0].match(n)

        # store the reference frame
        self.i = (self.__buildVersor__(j, k) if i is None else i).normalize()
        self.j = (self.__buildVersor__(k, i) if j is None else j).normalize()
        self.k = (self.__buildVersor__(i, j) if k is None else k).normalize()
                     


    def __buildVersor__(self, p1, p2):
        """
        The method returns the coordinates in 3D space of the 3rd versor defining the reference frame.
    
        Input:
            p1:     (Vector)
                    a Vector defining a point along the first axis.

            p2:     (Vector)
                    a Vector defining a point along the second axis.

        Output:
            v:      (Vector)
                    the vector containing the coordinates of the versor defining the third axis.

        Procedure:
            1)  O is identified as the perpendicular interception of the lines passing through (p1) and (p2). O is 
                calculated as
                                        
                                                        O = p1 * t

                where t can be calculated via the equation:
            
                                              t = sum(p1 * p2) / sum(p1 ** 2)

            2)  "p1 - O" is used to define the versor representing the first axis of the rotated frame (i).
            3)  "p2 - O" is used to build the versor representing the second axis of the rotated frame (j).
            4)  "i" x "j" defines the third versor of the rotated frame (k).
        """

        # check the entered data
        ut.classcheck(p1, ['Vector'])
        ut.classcheck(p2, ['Vector'])
        p1.match(p2)
    
        # 1) get the O coordinates
        t = (p1 * p2.values).sum(1).values / np.sum((p1 ** 2).values, axis=1)
        O = p1 * np.vstack([t.flatten() for n in p1.columns]).T
    
        # 2) get "i"
        i = (p1 - O).normalize()

        # 3) get "j"
        j = (p2 - O).normalize()

        # 4) get "k"
        return i.cross(j)



    def align(self, V):
        """
        align the vector V to the current reference frame.

        Input:
            V:  (Vector)
                the object to be aligned.

        Output:
            R:  (Vector or Joint)
                the object aligned to the current joint
        """
        
        # get the inverse and transposed reference frame
        R = self.to_dict()
        R = {i: np.linalg.inv(R[i].values).T for i in R}
        
        # return the rotated (i.e. aligned) vector
        return V.rotateby(R)



    def copy(self):
        """
        create a copy of the current object.
        """
        return ReferenceFrame(self.i, self.j, self.k)



    def atIndex(self, i):
        """
        return a 3x3 pandas.DataFrame containing the reference frame coordinates at the index i.

        Input:
            i:  (int, float)
                the index of the reference frame to be returned.

        Output:
            M:  (pandas.DataFrame)
                a 3x3 pandas.DataFrame containing the reference frame coordinates.
        """

        # check the entered data
        ut.classcheck(i, ["int", "float"])

        # check if the index is in the reference frame
        assert i in self.i.index.to_numpy(), "'i' is out of the versors index."
        
        # return the dataframe
        i_df = DataFrame(self.i.df.loc[i]).T
        j_df = DataFrame(self.j.df.loc[i]).T
        k_df = DataFrame(self.k.df.loc[i]).T
        df = i_df.append(j_df, sort=False).append(k_df, sort=False)
        df.index = ['i', 'j', 'k']
        return df



    def atSample(self, i):
        """
        return a 3x3 pandas.DataFrame containing the reference frame coordinates at the sample i.

        Input:
            i:  (int)
                the sample of the reference frame to be returned.

        Output:
            M:  (pandas.DataFrame)
                a 3x3 pandas.DataFrame containing the reference frame coordinates.
        """

        # check the entered data
        ut.classcheck(i, ["int"])

        # check if the index is in the reference frame
        assert i >= 0 and i < self.i.shape[0], "'i' is out of the versors samples range."
        
        # return the dataframe
        df = self.i.df.iloc[i].append(self.j.df.iloc[i], sort=False).append(self.k.df.iloc[i], sort=False)
        df.index = ['i', 'j', 'k']
        return df



    def to_dict(self):
        """
        return a dict containing 3x3 pandas.DataFrame(s) for each index containing the reference frame coordinates at
        the corresponding sample.
        """
        mdf = self.i.df.append(self.j.df, sort=False).append(self.k.df, sort=False)
        mdf.index = MultiIndex.from_product([["i", "j", "k"], self.i.index.to_numpy()])
        return {i: mdf.loc[IndexSlice[:, i], :] for i in self.i.index.to_numpy()}
    


    def from_dict(self, D, xunit=""):
        """
        generate a ReferenceFrame object starting from a dict object.

        Input:
            D:      (dict)
                    The dict must have one key per sample/index and each key must contain a 3 x 3 pandas.DataFrame or
                    ndarray with each row defining each versor and each column representing the dimensions.

            xunit:  (dict)
                    the label defining the index unit of measure.

        Output:
            R:  (ReferenceFrame)
                the corresponding ReferenceFrame object.
        """

        # check the entered data
        ut.classcheck(D, ['dict'])
        dms = []
        cls = [] 
        for i in D:
            ut.classcheck(D[i], ['ndarray', 'DataFrame'])
            assert np.all([j == 3 for j in D[i].shape]), "All arguments of 'D' must be a 3 x 3 ndarray or DataFrame."
            if D[i].__class__.__name__ == "DataFrame":
                dms += [D[i].columns.to_numpy()]
                cls += [D[i].__class__.__name__]
            else:
                dms += [['X', 'Y', 'Z']]
                cls += ['ndarray']
        dms = np.unique(np.array(dms).flatten())
        cls = np.unique(np.array(cls).flatten())
        assert len(dms) == 3, "Dimensions in the matrices do not match."
        assert len(cls) == 1, "all matrices must be DataFrames or ndarrays."

        # build the versors
        i = np.vstack([np.atleast_2d(D[l][0] if cls[0] == "ndarray" else D[l].values[0]) for l in D]).T
        i = dt.Vector({l: v for l, v in zip(dms, i)}, np.array([l for l in D]), xunit, "", "Versor")
        j = np.vstack([np.atleast_2d(D[l][1] if cls[0] == "ndarray" else D[l].values[1]) for l in D]).T
        j = dt.Vector({l: v for l, v in zip(dms, j)}, np.array([l for l in D]), xunit, "", "Versor")
        k = np.vstack([np.atleast_2d(D[l][2] if cls[0] == "ndarray" else D[l].values[2]) for l in D]).T
        k = dt.Vector({l: v for l, v in zip(dms, k)}, np.array([l for l in D]), xunit, "", "Versor")

        # create the ReferenceFrame object
        return ReferenceFrame(i, j, k)
        


    def rotateby(self, R, postprod=True):
        """
        rotate the reference frame by the rotation matrix R.

        Input:
            R:          (3x3 ndarray or dict of 3x3 ndarrays)
                        the rotation matrix or a dict containing the rotation matrix to be applied for each index.

            postprod:   (bool)
                        should the rotation matrix be post-multiplied?

        Output:
            M:  (ReferenceFrame)
                a new ReferenceFrame instance with rotated versors.
        """

        # check the entered data
        ut.classcheck(R, ['ndarray', 'dict'])
        ut.classcheck(postprod, ['bool'])
        if R.__class__.__name__ == "ndarray":
            assert np.all([i == 3 for i in R.shape]), "'R' must be a 3x3 ndarray."
            R = {i: R for i in self.i.index.to_numpy()}
        else:
            txt = "R index not matching the ReferenceFrame object."
            assert np.all([i in self.i.index.to_numpy() for i in R.keys()]), txt
            assert len(R.keys()) == self.i.shape[0], txt
        
        # rotate the reference frame
        M = self.to_dict()
        for i in R:
            M[i] = M[i].dot(R[i]) if postprod else DataFrame(R[i].dot(M[i].values), index=['i', 'j', 'k'], columns=dms)

        # return the reference frame
        return ReferenceFrame.from_dict(M, self.i.xunit)