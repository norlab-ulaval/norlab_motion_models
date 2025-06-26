from norlab_motion_models.motion_models.kinematics.ideal_diff_drive_2D import IdealDiffDrive2D
from copy import deepcopy
import numpy as np

def test_jacobians():

    params = {"wheel_radius":0.3,
            "base_width":2.0,
            "wheel_radius_gain":1.0,
            "base_width_gain":1.0}
    mm = IdealDiffDrive2D(params)
    jacobian = mm.jacobian
    assert jacobian.shape == (3, 2), "Jacobian shape is incorrect"
    assert mm.effective_wheel_radius == 0.3, "Effective wheel radius is incorrect"
    assert mm.effective_basewidth == 2.0, "Effective base width is incorrect"
    
    assert  0.0 == np.sum(mm.jacobian - np.array([[0.15, 0.15],
                                                  [0.0, 0.0],
                                                  [-0.25, 0.25]])), "Jacobian values are incorrect"
#
#def test_wheel_radius_setter():
#
#    params = {"wheel_radius":-0.3,
#            "base_width":2.0,
#            "wheel_radius_gain":1.0,
#            "base_width_gain":1.0}
#    mm = IdealDiffDrive2D(params)
#
#def test_wheel_radius_gain_setter():
#
#    params = {"wheel_radius":0.3,
#            "base_width":2.0,
#            "wheel_radius_gain":-1.0,
#            "base_width_gain":1.0}
#    mm = IdealDiffDrive2D(params)
#
#def test_base_width():
#
#    params = {"wheel_radius":0.3,
#            "base_width":-2.0,
#            "wheel_radius_gain":1.0,
#            "base_width_gain":1.0}
#    mm = IdealDiffDrive2D(params)
#def test_base_width_gain():
#
#    params = {"wheel_radius":0.3,
#            "base_width":2.0,
#            "wheel_radius_gain":1.0,
#            "base_width_gain":-1.0}
#    mm = IdealDiffDrive2D(params)
#

