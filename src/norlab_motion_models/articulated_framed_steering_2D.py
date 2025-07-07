from norlab_motion_models.kinematic_motion_model_2D import KinematicMotionModel
import numpy as np
DEBUG=True  # Set to True to enable debug prints
class ArticulatedFrameSteering(KinematicMotionModel):
    """
    Ideal differential drive kinematic model in 2D.
    This model assumes a differential drive robot with two wheels and no slip.
    """
    def __init__(self,params):
        super().__init__()
        self.name = "ArticulatedFrameSteering"
        self.state_dim = 3
        
        self.input_dim = 2  # [w_l, w_r] (left and right wheel speeds)
        self.control_input_frame = "joints"  # Control inputs are in the joint space (wheel speeds)
        
        self._maximum_param_value = 1000
        self._param_list = ["wheel_radius", "base_width", "base_width_gain", "wheel_radius_gain"]
        
        self._front_dist_joint_axle = 0.1
        self._rear_dist_joint_axle = 1.0
        self._front_dist_joint_axle_gain = 1.0
        self._rear_dist_joint_axle_gain = 1.0
        self._wheel_radius = 0.1  # Default wheel radius
        self._wheel_radius_gain = 1.0  # Gain for wheel radius
        
        self._effective_front_dist_joint_axle = self._front_dist_joint_axle * self._front_dist_joint_axle_gain
        self._effective_rear_dist_joint_axle = self._rear_dist_joint_axle * self._rear_dist_joint_axle_gain
        self._effective_wheel_radius = self._wheel_radius * self._wheel_radius_gain

        self._jacobian = None
        self.nb_group_state = 2
        self.load_params(params)
        self.compute_jacobian()

    ##### Start wheel radius properties #####
    @property
    def wheel_radius(self): 
        return self._wheel_radius

    @wheel_radius.setter
    def wheel_radius(self, value):
        try:
            assert 0.0 <value
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
            assert 0.0 <value
        except AssertionError:
            raise ValueError("Wheel radius gain must be greater than 0.0")
        self._wheel_radius_gain = value
        self.compute_jacobian()
    ##### End wheel radius properties #####

    ##### Start dist_joint_axle properties#####
    @property
    def front_dist_joint_axle(self): 
        return self._front_dist_joint_axle

    @front_dist_joint_axle.setter
    def front_dist_joint_axle(self, value):
        try:
            assert 0.0 <value
        except AssertionError:
            raise ValueError("Front_dist_joint_axle must be greater than 0.0") 
        self._front_dist_joint_axle = value
        self.compute_jacobian()

    @property
    def front_dist_joint_axle_gain(self):
        
        return self._front_dist_joint_axle_gain

    @front_dist_joint_axle_gain.setter
    def front_dist_joint_axle_gain(self, value):
        try:
            assert 0.0 <value
        except AssertionError:
            raise ValueError("Front_dist_joint_axle_gain must be greater than 0.0")
        self.front_dist_joint_axle_gain = value
        self.compute_jacobian()

    #### End front axle distance to joint properties #####

    @property
    def rear_dist_joint_axle(self):
        
        return self._rear_dist_joint_axle

    @rear_dist_joint_axle.setter
    def rear_dist_joint_axle(self, value):
        try:
            assert 0.0 <value
        except AssertionError:
            raise ValueError("Rear_dist_joint_axle must be greater than 0.0")
        
        self._rear_dist_joint_axle = value
        self.compute_jacobian()

    @property
    def rear_dist_joint_axle_gain(self):
        return self._rear_dist_joint_axle_gain

    @rear_dist_joint_axle_gain.setter
    def rear_dist_joint_axle_gain(self, value):
        try:
            assert 0.0 <value
        except AssertionError:
            raise ValueError("Rear_dist_joint_axle_gain must be greater than 0.0")
        self._rear_dist_joint_axle_gain = value
        self.compute_jacobian()

    #### End rear axle distance to joint properties #####

    def compute_jacobian(self):
        # Must compute the effective distance
        # Must compute the effective wheel radius
        # Must compute the Jacobian matrix J
        # Must compute the Jacobian inverse J_inv

        # Dummy example calculation
        #self._effective_basewidth = self._base_width * self._base_width_gain
        #self._effective_wheel_radius = self._wheel_radius * self._wheel_radius_gain
        #j_2x2 =  self._effective_wheel_radius * np.array([[1/2, 1/2],
        #                   [-1/(self._effective_basewidth), 1/(self._effective_basewidth)]])
        #j_2x2_inv = np.linalg.inv(j_2x2)
        #self._jacobian = np.array([[j_2x2[0,0],j_2x2[0,1]],
        #                    [0,0],
        #                    [j_2x2[1,0],j_2x2[1,1]]])
        #self._jacobian_inv = np.array([[j_2x2_inv[0,0],0, j_2x2_inv[0,1]],
        #                       [j_2x2_inv[1,0],0, j_2x2_inv[1,1]]])
        raise NotImplementedError("Jacobian computation is not implemented in this class. Please implement it in a subclass.")  
        if DEBUG:
            print("Jacobian computed:")
            print(self._jacobian)
            print("Jacobian inverse computed:")
            print(self._jacobian_inv)
            print("Effective wheel radius:")
            print(self._effective_wheel_radius)
            print("Effective base width:")
            print(self._effective_basewidth)

    def load_params(self,params):
        for key, value in params.items():
            if key in self._param_list:
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter '{key}' is not a valid parameter. Valid parameters are: {self._param_list}")

        # Replace with your actual Jacobian computation
        print(f"Recomputing Jacobian with:\n"
            f"  wheel_radius={self.wheel_radius}\n"
            f"  wheel_radius_gain={self.wheel_radius_gain}\n"
            f"  base_width={self.base_width}\n"
            f"  base_width_gain={self.base_width_gain}")


