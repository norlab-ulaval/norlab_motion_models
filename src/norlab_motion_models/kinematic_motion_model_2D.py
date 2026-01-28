from dataclasses import dataclass
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.spatial.transform import Rotation as R
from shapely.ops import nearest_points

#
State = np.ndarray
"""6x1 (x, y, z, RPY) state vector."""

PredictedStates = np.ndarray
"""9xN, N is the horizon in number of steps"""

ControlInput = np.ndarray
"""2xN, N is the horizon in number of steps"""

DeltaTimeVector = np.ndarray
"""1xN, N is the horizon in number of steps"""

@dataclass
class KinematicMotionModelParam(TypedDict):
    minimum_longitudinal_speed: float
    maximum_longitudinal_speed: float
    minimum_angular_speed: float
    maximum_angular_speed: float
    control_input_frame: str
    clip_out_of_bounds_commands: bool
    
    
    

class KinematicMotionModel:
    """
    Abstract base class for kinematic motion models.
    All kinematic motion models should inherit from this class and implement the `predict` method.
    """

    def __init__(self,params: KinematicMotionModelParam = None):

        self._jacobian = None  # Jacobian matrix for the kinematic model
        self.state_dim = 0  # Dimension of the state space
        self.input_dim = 0  # Dimension of the control input space
        self.name = "KinematicMotionModel"  # Name of the motion model
        self._jacobian_inv = None  # Inverse of the Jacobian matrix, if applicable
        self.control_input_frame = "body"  # "body", "joints"
        self.nb_group_state = 1  # Number of groups in the state, used for multi-group systems
        self._param_list = []
        self.params = params
        self.cmd_bounds_body_frame = None
        self.cmd_bounds_wheel_frame = None

    # def compute_jacobian(self):
    #    """
    #    Define the Jacobian matrix for the kinematic motion model and its inverse
    #    This method should be implemented by subclasses to define the specific Jacobian.
    #    """
    #    raise NotImplementedError("Subclasses must implement this method to define the Jacobian matrix.")
    
    def plot_polygon(self, ax, poly, color, label, alpha=0.5):
        """Utility to plot a shapely polygon"""
        y, x = poly.exterior.xy
        ax.fill(x, y, color=color, alpha=alpha, label=label)
        ax.plot(x, y, color=color)

    def compute_control_bouds(self,debug=False):
        """
        Define the control input bounds for the kinematic motion model.
        This method should be implemented by subclasses to define specific control bounds.
        """
        body_vels = shapely.geometry.box(
            self.params["minimum_longitudinal_speed"],
            self.params["minimum_angular_speed"],
            self.params["maximum_longitudinal_speed"],
            self.params["maximum_angular_speed"],
        )
        maximum_wheel_speed = np.max(self._jacobian_inv @ np.array(
            [
                [self.params["maximum_longitudinal_speed"]],
                [0.0],
                [0.0],
            ]
        )
        )
        
        
        
        maximum_wheel_speed_bounds = shapely.geometry.box(
            -maximum_wheel_speed,
            -maximum_wheel_speed,
            maximum_wheel_speed,
            maximum_wheel_speed,
        )
        
        max_wheel_speed_in_body_frame = self._jacobian @ np.array(maximum_wheel_speed_bounds.exterior.coords).T
        shapely_wheel_bound_in_body_frame= shapely.Polygon(np.column_stack((max_wheel_speed_in_body_frame[0,:], max_wheel_speed_in_body_frame[2,:])))

        intersect_bounds_body_frame = body_vels.intersection(shapely_wheel_bound_in_body_frame)
        coords = np.array(intersect_bounds_body_frame.exterior.coords).T
        
        print(coords.shape)
        print(np.vstack((coords[0,:], np.zeros(coords.shape[1]), coords[1,:])))
        intersect_bounds_wheel_frame = self._jacobian_inv @ np.vstack((coords[0,:], np.zeros(coords.shape[1]), coords[1,:]))
        
        intersect_bounds_wheel_frame = shapely.Polygon(intersect_bounds_wheel_frame.T)
        
        if intersect_bounds_body_frame.is_empty:
            raise ValueError("The intersection of body velocity bounds and wheel speed bounds is empty. Please check the parameters.")
        

        if debug:
            print("Body velocity bounds:", body_vels)
            print("Wheel speed bounds in body frame:", shapely_wheel_bound_in_body_frame)
            print("Intersection bounds:", intersect_bounds_body_frame)
            # --- Plot ---
            fig, ax = plt.subplots()

            self.plot_polygon(ax, body_vels, color="blue", label="Max body and ang speed")
            self.plot_polygon(ax, shapely_wheel_bound_in_body_frame, color="green", label="Max wheel speed from max lin speed")
            self.plot_polygon(ax, intersect_bounds_body_frame, color="Orange", label="Intersection", alpha=0.8)

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Cmd angular speed")
            ax.set_ylabel("Cmd longitudinal speed")
            ax.legend()
            ax.set_title("Cmd space intersection")
            ax.set_aspect("equal")
            plt.show()


        self.cmd_bounds_body_frame = intersect_bounds_body_frame
        self.cmd_bounds_wheel_frame = intersect_bounds_wheel_frame

        


    def validate_and_clip_commands(self, commands: ControlInput) -> ControlInput:
        """
        Validate and clip the control commands to be within the defined bounds.
        :param commands: Control input commands to validate and clip.
        :return: Clipped control input commands.
        """

        if self.control_input_frame == "body":
            self.cmd_space = self.cmd_bounds_body_frame
            index= [0,2]
        elif self.control_input_frame == "joints":
            self.cmd_space = self.cmd_bounds_wheel_frame
            index= [0,1]
        else:
            raise ValueError("control_input_frame must be either 'joints' or 'body'.")
        
        if self.cmd_bounds_body_frame is None or self.cmd_bounds_wheel_frame is None:
            raise ValueError("Control bounds have not been computed. Please call compute_control_bounds() first.")

            
        clipped_commands = np.copy(commands)

        for i in range(commands.shape[1]):
            command_point_body = shapely.geometry.Point(commands[index[0], i], commands[index[1], i])
            print("cmd space",self.cmd_space)
            if not self.cmd_space.contains(command_point_body):
                clipped_point_body = nearest_points( self.cmd_space, command_point_body)[0] 
                print(f"Command {command_point_body} is out of bounds. Clipping to {clipped_point_body}.")
                clipped_commands[index[0], i] = clipped_point_body.x
                clipped_commands[index[1], i] = clipped_point_body.y

        
        return clipped_commands




        
    def integrate_position(self, initial_state: State, body_commands, dt):
        """
        Integrate the position based on the current state, control input, and time step.
        Return the position in the initial state frame.
        :param initial_state: Current state of the system.
        :param speed_state: speed_state (body velocity) to apply.
        :param dt: Time step for the integration.
        :return: Updated state after applying the control input.
        """
        print("initial state", initial_state)
        if self._jacobian is None:
            raise ValueError("Jacobian matrix J is not initialized.")

        if initial_state.shape[0] != self.state_dim:
            raise ValueError("Initial state dimension does not match the state dimension of the model.")

        if body_commands.shape[0] != 3:
            print("body_commands.shape", body_commands.shape)
            raise ValueError("Control input dimension does not match the control dimension of the model.")

        rotation = R.from_euler("z", initial_state[2, 0], degrees=False)
        transform_global = np.eye(3)
        transform_global[:2, :2] = rotation.as_matrix()[:2, :2]
        transform_global[:2, 2] = initial_state[:2, 0]

        predicted_state = np.zeros((self.state_dim, dt.shape[0]))

        for i in range(body_commands.shape[1]):

            command = body_commands[0:, i]
            # Compute the change in state using the Jacobian
            delta_state = command * dt[i]

            rotation = R.from_euler("z", delta_state[2], degrees=False)
            transform_delta = np.eye(3)
            transform_delta[:2, :2] = rotation.as_matrix()[:2, :2]
            transform_delta[:2, 2] = delta_state[:2]  # Add positions

            transform_global = transform_global @ transform_delta
            theta = np.arctan2(transform_global[1, 0], transform_global[0, 0])

            predicted_state[2, i] = theta
            predicted_state[:2, i] = transform_global[:2, 2]
        # print("predicted_state", predicted_state)
        return predicted_state

    def predict(self, initial_state: State, control_input: ControlInput, dt: DeltaTimeVector) -> PredictedStates:
        """Compute the next state based on the current state, control input, and time step.
        This method uses the Jacobian matrix to compute the change in state due to the control input.
        :param state assume a 3dof full state vector (x, y, theta, x_dot, y_dot, theta_dot, x_ddot, y_ddot, theta_ddot)
        :param control_input assume a active joint state (x_dot, y_dot, theta_dot)
        :param dt: assume a vector of dt.

        Args:
            state (_type_): _description_
            control_input (_type_): _description_
            dt (_type_): _description_
        """

        if isinstance(dt, float) or isinstance(dt, int):
            dt = np.ones(control_input.shape[1]) * dt
        N = dt.shape[0]

        initial_shape = (9, 1)
        if initial_state.shape != initial_shape:
            raise ValueError(f"inital_state shape should be {initial_shape}, not {initial_state.shape}")

        control_shape = (self.input_dim, N)
        if control_input.shape != control_shape:
            raise ValueError(f"control_input shape should be ({control_shape}), not {control_input.shape}")

        
        #if dt.shape != (1, N) or dt.shape != (N,):
        #    raise ValueError(f"dt shape should be ({dt.shape}), not {dt.shape}")
        if self.params["clip_out_of_bounds_commands"]:
                control_input = self.validate_and_clip_commands(control_input)

        if self.control_input_frame == "joints":
            ## Assuming the control input is in the joint_state
            speed_state = self.compute_fwd_kinematics(control_input)  # 3xN command

            
        elif self.control_input_frame == "body":
            ## Assuming the control input is in the body frame
            speed_state = control_input
        else:
            raise ValueError("control_input_frame must be either 'joints' or 'body'.")

        print("twist_msg", speed_state)

        integrated_state = self.integrate_position(initial_state, speed_state, dt)

        return integrated_state

    def compute_fwd_kinematics(self, joint_state):
        """Compute forward kinematics for a given joint state."""

        if self._jacobian is None:
            raise ValueError("Jacobian matrix J is not initialized.")
        if joint_state.shape[0] != self._jacobian.shape[1]:
            raise ValueError("Joint state dimension does not match the Jacobian matrix dimension.")

        return self._jacobian @ joint_state

    def compute_inv_kinematics(self, end_effector_state):
        """Compute reverse kinematics for a given end-effector state."""

        if self._jacobian is None:
            raise ValueError("Jacobian matrix J is not initialized.")
        if end_effector_state.shape[0] != self._jacobian.shape[0]:
            raise ValueError("End-effector state dimension does not match the Jacobian matrix dimension.")

        return self._jacobian.T @ end_effector_state

    def load_params(self, params):
        for key, value in params.items():
            if key in self._param_list:
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Parameter '{key}' is not a valid parameter. Valid parameters are: {self._param_list}"
                )

    def save_params(self):
        """
        Return a dictionary of safe parameters for the kinematic motion model.
        """
        raise NotImplementedError()
        raise NotImplementedError()
