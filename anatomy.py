# generic imports

import numpy as np
import pyomech.utils as ut
import pyomech.vectors as dt


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
    def __init__(self, O, I, P):
        """
        Input:
            O:  (Vector)
                the location of the joint centre.

            I:  (Vector)
                A vector reflecting the orientation of the first axis of the joint reference frame with respect to the
                external reference frame.

            P:  (Vector)
                A vector reflecting a point passing through the second axis of the joint reference frame with respect
                to the external reference frame. This point serves to define the second axis orientation.

        Output:
            O:  (Vector)
                The centre of the joint location as provided.

            R:  (dict of ndarrays)
                a dict having the indices of O as keys and, for each key, a n-dimensional matrix defining the rotations
                necessary to translate the external reference frame into the joint reference frame.

        Note:
            during the generation of a Joint object, the I and J axes are used to build up the joint reference frame.
            The third axis (K) is calculated via cross product between the versors i and j respectively extracted from
            I and J.
        """

        # check the entered data
        ut.classcheck(O, ['Vector'])
        ut.classcheck(I, ['Vector'])
        ut.classcheck(P, ['Vector'])
        O.match(I)
        O.match(P)

        # store the joint centre
        self.O = O

        # get the rotation matrix necessary to define the joint reference frame
        self.R = __getR__(I, P, False)[0]



    # get a rotation matrix from a vector and a point in space.
    def __getR__(v, p, rotate_frame=False):
        """
        The method calculates a rotation matrix starting from a vector and a point in the space.
    
        Input:
            v : (Vector)
                a 3D Vector defining a vector originating from the origin of the reference frame.

            p : (Vector)
                a 3D Vector being part of the same reference frame of v and not being part of the line passing through
                v.

            rotate_frame : (Bool)
                if False (default) the rotation matrix returned by this method reflect the rotation bringing the
                reference frame defined by "v" and "p" aligned to the external reference frame:
                [[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]]
                If True, the rotation matrix will reflect the rotation necessary to align the same external reference
                frame to the "v"-"p" reference frame.

        Output:
            R: (dict)
                a dict with each key defining one sample of "v" and "p". For each sample, a 3x3 ndarray in the form
                Q x J where Q are the dimensions of "v" and "p" and "J" are the names of the new reference frame axes.

            O: (Vector)
                a pybiomech.vectors.Vector object defining the coordinates of the origin aroung which the rotation
                matrix is calculated. The origin coordinates will reflect its distance from the origin of "v".

        Notes:
            The vector is assumed to be the first axis of the rotated frame and the point will define the equations of
            the line defining the versor reflecting the second axis of the rotated frame.

        Procedure:
            For each time sample of "v" and "p" the following steps are made:
            1)  O is identified along "v" as the point of interception between the line passing through "v" and
                parallel to it with the plane "h" orthogonal to "v" and passing trough "p". O is calculated as
                                        
                                                    O = v * t

                where t can be calculated via the equation:
            
                                            t = sum(p * v) / sum(v ** 2)

            2)  "v" is used to define the versor representing the first axis of the rotated frame (i).
            3)  "p-O" is used to build the versor representing the second axis of the rotated frame (j).
            4)  "i" x "j" defines the third versor of the rotated frame (k).
            5)  R is calculated as the inverse of the transposed [i, j, k] 3x3 matrix.
            6)  if the frame has to be rotated rather than the vector, the rotation matrices calculated for each time 
                sample is transposed.
        """

        # check the entered data
        ut.classcheck(v, ['Vector'])
        ut.classcheck(p, ['Vector'])
        ut.classcheck(rotate_frame, ['bool'])
        assert v.match(p), "v and p must have same dimensions and sample size."

        # 1) get the O coordinates
        t = np.atleast_2d((p * v).sum(1).values.flatten() / np.sum((v ** 2).values, axis=1).flatten())
        O = v * np.vstack([t for n in v.columns]).T
    
        # 2) get "i"
        i = v.normalize()

        # 3) get "j"
        j = (p - O).normalize()

        # 4) get "k"
        k = i.cross(j)

        # 5) get "R"
        R = {}
        for n, q in enumerate(v.index.to_numpy()):
            R[q] = np.linalg.inv(np.float64(np.vstack(np.atleast_2d(i.values[n], j.values[n], k.values[n])).T))

        # 6) if the the reference frame has to be rotated, transpose the rotation matrices
        R = {q: R[q].T for q in R} if rotate_frame else R
    
        # return the calculated data
        return R, O
