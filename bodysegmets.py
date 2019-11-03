# generic imports

import numpy as np
from utils import classcheck
from vectors import Vector


"""
The following coefficients corresponds to those described by de Leva (1996a) in his Table 4 and are used to estimate
the mass, the Centre of Mass (CoM) position and the gyration radius around each rotation axis.

Reference:
    de Leva P. (1996a) Adjustments to Zatiorsky-Seluyanov's segment inertia parameters. Journal of Biomechanics,
        29(9):1223-30
"""

segments = {

    # Head segment is calculated from the head vertex to the spinous process of the 7th cervical vertebrae.
    'Head': {
        'male':{
            'mass': 0.694,
            'CoM': 0.5002,
            'gyration_radius_ap': 0.303,
            'gyration_radius_ml': 0.315,
            'gyration_radius_vt': 0.261
            },
        'female':{
            'mass': 0.668,
            'CoM': 0.4841,
            'gyration_radius_ap': 0.271,
            'gyration_radius_ml': 0.295,
            'gyration_radius_vt': 0.263
            }
        },

    # Trunk segment is calculated from the 7th cervical vertebrae and the mid-point between the hips joint center.
    'Trunk': {
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
    'Arm': {
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
    'Forearm': {
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
    'Hand': {
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
    'Thigh': {
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
    'Shank': {
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
    'Foot': {
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

    # The whole body "segment" is calculated from the spinous process of the 7th cervical vertebrae to the hips
    # mid-point.
    'Whole-body': {
        'male':{
            'mass': 1.,
            'CoM': ,
            'gyration_radius_ap': ,
            'gyration_radius_ml': ,
            'gyration_radius_vt': 
            },
        'female':{
            'mass': 1.,
            'CoM': ,
            'gyration_radius_ap': ,
            'gyration_radius_ml': ,
            'gyration_radius_vt': 
            }
        }
    }



class BodySegment():
    """
    A class representing the position in space of a body segment. It provides both the inertial and center of mass
    parameters according to the equation of de Leva (1996).
    """


    # dependancies
    import numpy as np
    from utils import classcheck


    # constructor
    def __init__(self, origin, end, height, weight, male, what):
        """
        Input:
            origin: (3D Vector)
                    the vector with the data defining the position of the origin of the body segment.

            end:    (3D Vector)
                    the vector with the data defining the position of the end of the body segment.

            height: (float)
                    the height of the participant in meters.

            weight: (float)
                    the weight of the participant in kg.

            male:   (bool)
                    True if the participant is a male, False otherwise.

            what:   (str)
                    any of ["Head", "Trunk", "Arm", "Forearm", "Thigh", "Shank", "Foot"].
        """
        
        # Check the entered parameters
        for i in [origin, end]:
            classcheck(i, ['Vector'])
            assert i.shape[1] == 3, "'origin' and 'end' must be a 3D vector."
        assert np.all([i in origin.df.columns] for i in end.df.columns), "'origin' and 'end' must have the same ndim."
        same_index = np.sum(np.diff(origin.index.to_numpy() - end.index.to_numpy())) == 0
        assert same_index, "'origin' and 'end' must have same index."
        classcheck(height, ['float', 'int'])
        classcheck(weight, ['float', 'int'])
        assert male or not male, "'male' must be a boolean."
        txt = "'what' must by any of the following string: " + [i for i in segments.keys()]
        assert what.lower() in [i for i in segments.keys()], txt

        # set the entered values
        self.origin = origin
        self.end = end
        self.height = height
        self.weight = weight
        self.gender = 'male' if male else 'female'
        self.what = what
        self.length = (origin - end).module()

        # get the specific parameters for the current segment according to de Leva (1996)
        self.mass =  self.weight * segments[what][self.gender]['mass']
        self.CoM = (end - origin) * segments[what][self.gender]['CoM'] + origin
        I = {
            'AP': (self.length.values.flatten() * segments[what][self.gender]['gyration_radius_ap']) ** 2 * self.mass,
            'ML': (self.length.values.flatten() * segments[what][self.gender]['gyration_radius_ml']) ** 2 * self.mass,
            'VT': (self.length.values.flatten() * segments[what][self.gender]['gyration_radius_ml']) ** 2 * self.mass
            }
        idx = self.length.index.to_numpy()
        self.inertia = Vector(I, idx, self.length.xunit, self.length.dunit, 'Moment of inertia')



"""
The following coefficients corresponds to those described by de Leva (1996b) in his Table 2 and are used to estimate
the longitudinal distances of different joint centes according to external markers.

Reference:
    de Leva P. (1996b) Joint center longitudinal positions computed from a selected subset of Chandler's data.
        Journal of Biomechanics, 29(9):1231-33
"""

joints = {

    # The shoulder joint centre is calculated from the acromion to the radial tubercle of the same side.
    'Shoulder': 0.104,

    # The elbow joint centre is calculated from the acromion to the radial tubercle of the same side.
    'Elbow': 0.957,
    
    # The wrist joint centre is calculated from the radial tubercle to the styloid process of the radius.
    'Wrist': 1.006,

    # The hip joint centre is calculated from the great throcanter to the tibial process.
    'Hip': -0.007,

    # The knee joint centre is calculated from the great throcanter to the tibial process.
    'Knee': 0.926,

    # The ankle joint centre is calculated from the tibial process to the lateral malleoulus.
    'Ankle': 1.016,
    }



def get_jointcentre(origin, end, what):
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
        classcheck(i, ['Vector'])
        assert i.shape[1] == 3, "'origin' and 'end' must be a 3D vector."
    assert np.all([i in origin.df.columns] for i in end.df.columns), "'origin' and 'end' must have the same ndim."
    same_index = np.sum(np.diff(origin.index.to_numpy() - end.index.to_numpy())) == 0
    assert same_index, "'origin' and 'end' must have same index."
    txt = "'what' must by any of the following string: " + [i for i in joints.keys()]
    assert what.lower() in [i for i in joints.keys()], txt
    
    # return the joint centre
    return (end - origin) * joints[what] + origin