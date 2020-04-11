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



def deLeva_jointcentre(origin, end, what):
    """
    this method returns a pyomech.Vector object representing the requried joint centre according to de Leva (1996b).

    Input:
        origin: (3D Vector)
                the vector with the data defining the position of the origin of the body segment.

        end:    (3D Vector)
                the vector with the data defining the position of the end of the body segment.

        what:   (str)
                any of ["Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle"].
    
    Output:
        joint:  (3D Vector)
                a 3D Vector defining the location of the joint centre.
    """
    
    # Check the entered parameters
    for i in [origin, end]:
        ut.classcheck(i, ['Vector'])
        assert i.shape[1] == 3, "'origin' and 'end' must be a 3D vector."
    assert np.all([i in origin.df.columns] for i in end.df.columns), "'origin' and 'end' must have the same ndim."
    same_index = np.sum(np.diff(origin.index.to_numpy() - end.index.to_numpy())) == 0
    assert same_index, "'origin' and 'end' must have same index."
    txt = "'what' must by any of the following string: " + [i for i in joints.keys()]
    assert what.lower() in [i for i in joints.keys()], txt
    
    # return the joint centre
    return (end - origin) * joints[what] + origin



# Joint class
class Joint():
    """
    A class representing a joint with the location of its centre and a 3D reference frame aligned with the joint.
    """

    # constructor
    def __init__(self, O, i=None, j=None ,k=None):
        """
        Input:
            O:  (Vector)
                the location of the joint centre.

            i:  (Vector, None)
                A vector reflecting the orientation of the versor along the first axis of the joint reference frame.

            j:  (Vector, None)
                A vector reflecting the orientation of the versor along the second axis of the joint reference frame.

            k:  (Vector, None)
                A vector reflecting the orientation of the versor along the third axis of the joint reference frame.

        Output:
            O:  (Vector)
                The centre of the joint location as provided.

            referenceFrame:  (dict of ndarrays)
                             a ReferenceFrame class object.

        Note:
            a minimum of two versors between i, j and k must be provided.
        """

        # check the entered data
        ut.classcheck(O, ['Vector'])
        ut.classcheck(i, ['Vector', 'NoneType'])
        ut.classcheck(j, ['Vector', 'NoneType'])
        ut.classcheck(k, ['Vector', 'NoneType'])
        versors = [n for n in [i, j, k] if n is not None]
        for n in versors:
            O.match(n)

        # store the joint centre
        self.O = O

        # get the rotation matrix necessary to define the joint reference frame
        self.rf = ReferenceFrame(i, j, k)



    def align(self, V):
        """
        align V to the self reference frame.

        Input:
            V:  (Vector)
                the object to be aligned.

        Output:
            R:  (Vector or Joint)
                the object aligned to the current joint
        """

        # check the entered data
        ut.classcheck(V, ['Vector'])
        V.match(self.O)
            
        # return the rotated (i.e. aligned) vector
        return self.rf.align(V - self.O)
        


    def cartesianAngle(self):
        """
        calculate the orientation of the current Jointoint vector V with respect to the current joint reference frame.

        Input:
            V:          (Vector)
                        a Vector with the same shape of V.

            isAligned:  (bool)
                        if True, V is considered being part of the Joint reference frame. Otherwise, V is firstly
                        rotated to match with the reference frame of the current Joint.

        Output:
            A:          (Vector)
                        a Vector with dimensions defining the planes on which the angles are calculated. The angles
                        are provided in radiants.
        """

        # check the entered data
        ut.classcheck(isAligned, ['bool'])
        ut.classcheck(V, ['Vector'])
        assert np.all([i == v for i, v in zip(V.shape, self.O.shape)]), "'V' must have shape " + str(self.O.shape)
        assert np.all([i in self.O.index.to_numpy() for i in V.index.to_numpy()]), "'V' index does not match with 'O'."

        # check if V has to be aligned with the Joint reference frame
        Vrot = V.copy() if isAligned else self.align(V)

        # get the planes along which calculating the angles
        planes = [i for i in combinations(Vrot.columns.to_numpy(), 2)]

        # for each plane get the angle
        A = {}
        for pl in planes:
            A["-".join(pl)] = np.arctan2(V[pl[1]].values.flatten(), V[pl[0]].values.flatten())
            correct = np.argwhere(A["-".join(pl)] < 0).flatten()
            if len(correct) > 0:
                A["-".join(pl)][correct] = 2 * np.pi + A["-".join(pl)][correct]
        return dt.Vector(A, self.O.index.to_numpy(), self.O.xunit, "rad", "Angle")



    def jointAngle(self, ref, order=[0, 1, 2]):
        """
        calculate the angle of the current Joint vs the entered reference joint.

        Input:
            ref:        (Joint)
                        a Joint with the same shape of self.

            order:      (list, ndarray)
                        the order of axes around which the aligning rotations have to be performed.

        Output:
            A:          (Vector)
                        a Vector with dimensions defining the planes on which the angles are calculated. The angles
                        are provided in radiants.
        """

        # check the entered data
        ut.classcheck(ref, ['Joint'])
        ut.classcheck(order, ['list', 'ndarray'])
        self.O.match(ref.O)
        assert len(order) == len(ref.O.columns), "Order must have length " + str(len(ref.O.columns))
        
        # get the string defining the order of the rotation
        ord = "".join(["xyz"[i] for i in order])

        # for each instant, get the rotation aligning the current joint to the original reference frame.
        # Next, multiply it by the ref reference frame
        A = np.atleast_2d([])
        for i in self.rf:
            R = Rotation.from_dcm(np.linalg.inv(self.rf[i]).dot(ref.reference[i])).as_euler(ord)
            if A.shape[1] == 0:
                A = np.atleast_2d(R)
            else:
                A = np.vstack([A, np.atleast_2d(R)])
        
        # return the angles
        A = {i: v for i, v in zip(self.O.columns.to_numpy()[order], A.T)}
        return dt.Vector(A, self.O.index.to_numpy(), self.O.xunit, "rad", "Joint angle")


    # copy operation
    def copy(self):
        """
        return an exact copy of self
        """
        p1 = {i: np.array([]) for i in self.O.columns.to_numpy()}
        p2 = {i: np.array([]) for i in self.O.columns.to_numpy()}
        for i in self.rf:
            for j, v in enumerate(self.O.columns.to_numpy()):
                p1[v] = np.append(p1[v], self.rf[i][0][j])
                p2[v] = np.append(p2[v], self.rf[i][1][j])
        p1 = dt.Vector(p1, self.O.index.to_numpy(), self.O.xunit, self.O.dunit, self.O.type)
        p2 = dt.Vector(p2, self.O.index.to_numpy(), self.O.xunit, self.O.dunit, self.O.type)
        return Joint(self.O.copy(), p1, p2)


def angle3Points(A, B ,C):
    """                 
    calculate the angle ABC between joints using the Carnot theorem.

    Input:
        A, B, C:    (Joint)
                    Joint objects with same shape

    Output: 
        C:          (Vector)
                    a vector with the angle in radiants for the calculated angle.
    """

    # check dependancies
    ut.classcheck(A, ['Joint'])
    ut.classcheck(B, ['Joint'])
    ut.classcheck(C, ['Joint'])
    Ax = A.O.index.to_numpy()
    Bx = B.O.index.to_numpy()
    Cx = C.O.index.to_numpy()
    Ay = A.O.columns.to_numpy()
    By = B.O.columns.to_numpy()
    Cy = C.O.columns.to_numpy()
    txt = "'A', 'B', 'C' must have same index and dimensions."
    assert np.all([i in Bx for i in Ax]), txt
    assert np.all([i in Cx for i in Bx]), txt
    assert np.all([i in Ax for i in Cx]), txt
    assert np.all([i in By for i in Ay]), txt
    assert np.all([i in Cy for i in By]), txt
    assert np.all([i in Ay for i in Cy]), txt

    # return the angle
    return B.O.angle(A.O, C.O)



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