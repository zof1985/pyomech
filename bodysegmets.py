# generic imports

import numpy as np
from utils import classcheck



# de Leva (1996) references

segments = {

    # Head segment is calculated from the head vertex to the spinous process of the 7th cervical vertebrae.
    'Head': {
        'male':{
            'mass': 0.694,
            'CoM': 0.5002,
            'rad_gyr_ap': 0.303,
            'rad_gyr_ml': 0.315,
            'rad_gyr_vt': 0.261
            },
        'female':{
            'mass': 0.668,
            'CoM': 0.4841,
            'rad_gyr_ap': 0.271,
            'rad_gyr_ml': 0.295,
            'rad_gyr_vt': 0.263
            }
        },

    # Trunk segment is calculated from the 7th cervical vertebrae and the mid-point between the hips joint center.
    'Trunk': {
        'male':{
            'mass': 0.4346,
            'CoM': 0.5138,
            'rad_gyr_ap': 0.328,
            'rad_gyr_ml': 0.306,
            'rad_gyr_vt': 0.169
            },
        'female':{
            'mass': 42.57,
            'CoM': 0.4964,
            'rad_gyr_ap': 0.307,
            'rad_gyr_ml': 0.292,
            'rad_gyr_vt': 0.147
            }
        },
    
    # The upper-arm segment is calculated from the Shoulder joint center to the elbow joint center of the same side.
    'Arm': {
        'male':{
            'mass': 0.0271,
            'CoM': 0.5772,
            'rad_gyr_ap': 0.285,
            'rad_gyr_ml': 0.269,
            'rad_gyr_vt': 0.158
            },
        'female':{
            'mass': 0.0255,
            'CoM': 0.5754,
            'rad_gyr_ap': 0.278,
            'rad_gyr_ml': 0.260,
            'rad_gyr_vt': 0.148
            }
        },
    
    # The forearm segment is calculated from the elbow joint center to the wrist joint center of the same side.
    'Forearm': {
        'male':{
            'mass': 0.0162,
            'CoM': 0.4574,
            'rad_gyr_ap': 0.276,
            'rad_gyr_ml': 0.265,
            'rad_gyr_vt': 0.121
            },
        'female':{
            'mass': 0.0138,
            'CoM': 0.4559,
            'rad_gyr_ap': 0.261,
            'rad_gyr_ml': 0.257,
            'rad_gyr_vt': 0.094
            }
        },
    
    # The hand segment is calculated from the wrist joint center to the end of the 3rd metacarpale of the same side.
    'Hand': {
        'male':{
            'mass': 0.061,
            'CoM': 0.7900,
            'rad_gyr_ap': 0.628,
            'rad_gyr_ml': 0.513,
            'rad_gyr_vt': 0.401
            },
        'female':{
            'mass': 0.056,
            'CoM': 0.7474,
            'rad_gyr_ap': 0.531,
            'rad_gyr_ml': 0.454,
            'rad_gyr_vt': 0.335
            }
        },
    
    # The thigh segment is calculated from the hip joint center to the knee joint center of the same side.
    'Thigh': {
        'male':{
            'mass': 0.1416,
            'CoM': 0.4095,
            'rad_gyr_ap': 0.329,
            'rad_gyr_ml': 0.329,
            'rad_gyr_vt': 0.149
            },
        'female':{
            'mass': 0.1478,
            'CoM': 0.3612,
            'rad_gyr_ap': 0.369,
            'rad_gyr_ml': 0.364,
            'rad_gyr_vt': 0.162
            }
        },
    
    # The shank segment is calculated from the knee joint center to the lateral malleolus of the same side.
    'Shank': {
        'male':{
            'mass': 0.0432,
            'CoM': 0.4459,
            'rad_gyr_ap': 0.255,
            'rad_gyr_ml': 0.249,
            'rad_gyr_vt': 0.103
            },
        'female':{
            'mass': 0.0481,
            'CoM': 0.4416,
            'rad_gyr_ap': 0.271,
            'rad_gyr_ml': 0.267,
            'rad_gyr_vt': 0.093
            }
        },
    
    # The foot segment is calculated from the heel to the tip of the longest toe of the foot.
    'Foot': {
        'male':{
            'mass': 0.0137,
            'CoM': 0.4415,
            'rad_gyr_ap': 0.257,
            'rad_gyr_ml': 0.245,
            'rad_gyr_vt': 0.124
            },
        'female':{
            'mass': 0.0129,
            'CoM': 0.4014,
            'rad_gyr_ap': 0.299,
            'rad_gyr_ml': 0.279,
            'rad_gyr_vt': 0.139
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
        for i in segments[what][self.gender]:
            if i != 'Mass':
                setattr(self, i, (end - origin) * segments[what][self.gender][i] + origin)
            else:
                setattr(self, i, self.weight * segments[what][self.gender][i])
