from typing import TypedDict

import numpy as np
from kinematic_motion_model_2D import (KinematicMotionModel,
                                       KinematicMotionModelParam)

DEBUG = True  # Set to True to enable debug prints

from dataclasses import dataclass


@dataclass
class IdealDiffDrive2DParam(KinematicMotionModelParam):
    wheel_radius: float
    base_width: float
    wheel_radius_gain: float
    base_width_gain: float
    
    


class IdealDiffDrive2D(KinematicMotionModel):
    """
    Ideal differential drive kinematic model in 2D.
    This model assumes a differential drive robot with two wheels and no slip.
    """

    def __init__(self, params: IdealDiffDrive2DParam, verbose=True):
        self.verbose = verbose
        super().__init__(params=params)
        self.name = "IdealDiffDrive2D"
        self.state_dim = 9  # [x, y, theta]
        
        self.control_input_frame = params["control_input_frame"]  # Control inputs are in the joint space (wheel speeds)
        self._maximum_param_value = 1000
        self._param_list = list(IdealDiffDrive2DParam.__required_keys__)
        self._wheel_radius = 0.1
        self._wheel_radius_gain = 1.0
        self._base_width = 0.5
        self._base_width_gain = 1.0
        self._effective_wheel_radius = None
        self._effective_basewidth = None
        self._jacobian = None

        self.load_params(params)
        self.compute_jacobian()

        if self.verbose:
            print("Jacobian computed:")
            print(self._jacobian)
            print("Jacobian inverse computed:")
            print(self._jacobian_inv)
            print("Effective wheel radius:")
            print(self._effective_wheel_radius)
            print("Effective base width:")
            print(self._effective_basewidth)

        print(
            f"Motion Model params:\n"
            f"  wheel_radius={self.wheel_radius}\n"
            f"  wheel_radius_gain={self.wheel_radius_gain}\n"
            f"  base_width={self.base_width}\n"
            f"  base_width_gain={self.base_width_gain}"
        )

        if self.control_input_frame == "joints":
            print("  control_input_frame=joints (wheel speeds)")
            self.input_dim = 2  # [w_l, w_r] (left and right wheel speeds)
        elif self.control_input_frame == "body":
            print("  control_input_frame=body (linear and angular velocities)")
            self.input_dim = 3  # [v, omega] (linear and angular velocities)
    @property
    def wheel_radius(self):
        return self._wheel_radius

    @wheel_radius.setter
    def wheel_radius(self, value):
        try:
            assert 0.0 < value
        except AssertionError:
            raise ValueError("Wheel radius must be greater than 0.0")
        self._wheel_radius = value
        self.compute_jacobian()

    @property
    def wheel_radius_gain(self):

        return self._wheel_radius_gain

    @wheel_radius_gain.setter
    def wheel_radius_gain(self, value):
        try:
            assert 0.0 < value
        except AssertionError:
            raise ValueError("Wheel radius gain must be greater than 0.0")
        self._wheel_radius_gain = value
        self.compute_jacobian()

    @property
    def base_width(self):

        return self._base_width

    @base_width.setter
    def base_width(self, value):
        try:
            assert 0.0 < value
        except AssertionError:
            raise ValueError("Base width must be greater than 0.0")

        self._base_width = value
        self.compute_jacobian()

    @property
    def base_width_gain(self):
        return self._base_width_gain

    @base_width_gain.setter
    def base_width_gain(self, value):
        try:
            assert 0.0 < value
        except AssertionError:
            raise ValueError("Base width gain must be greater than 0.0")
        self._base_width_gain = value
        self.compute_jacobian()

    def compute_jacobian(self):

        # Dummy example calculation
        self._effective_basewidth = self._base_width * self._base_width_gain
        self._effective_wheel_radius = self._wheel_radius * self._wheel_radius_gain

        j_2x2 = self._effective_wheel_radius * np.array(
            [[1 / 2, 1 / 2], [-1 / (self._effective_basewidth), 1 / (self._effective_basewidth)]]
        )
        j_2x2_inv = np.linalg.inv(j_2x2)
        self._jacobian = np.array([[j_2x2[0, 0], j_2x2[0, 1]], [0, 0], [j_2x2[1, 0], j_2x2[1, 1]]])  # y  # Y)
        self._jacobian_inv = np.array([[j_2x2_inv[0, 0], 0, j_2x2_inv[0, 1]], [j_2x2_inv[1, 0], 0, j_2x2_inv[1, 1]]])

        self.compute_control_bouds()
        


if __name__ =="__main__":
    params = IdealDiffDrive2DParam(
        wheel_radius=0.3,
        base_width=1.08,
        wheel_radius_gain=1.0,
        base_width_gain=1.0,
        minimum_longitudinal_speed=-5.0,
        maximum_longitudinal_speed=5.0,
        minimum_angular_speed=-4.0,
        maximum_angular_speed=4.0,
        control_input_frame="joints",
        clip_out_of_bounds_commands=True,
    )

    
    model = IdealDiffDrive2D(params=params, verbose=True)
    cmd = np.array([[0.0], [26.0]])
    cmd = np.hstack((cmd, cmd, cmd))
    

    print(model.predict(np.zeros((9,1)), cmd, dt=1.0))

    print(model.input_dim)
    #inv_cmd = model._jacobian_inv @cmd 
    #twist = model._jacobian @ inv_cmd
    #print("cmd",cmd)
    #print("wheel cmd",inv_cmd)
    #print("reverted_twist",twist)
    model.compute_control_bouds(debug=True)    #print("wheel cmd",inv_cmd)
    #print("reverted_twist",twist)
    model.compute_control_bouds(debug=True)