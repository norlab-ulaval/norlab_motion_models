from scipy.spatial.transform import Rotation as R
import numpy as np

#
State = np.ndarray
"""6x1 (x, y, z, RPY) state vector."""

PredictedStates = np.ndarray
"""9xN, N is the horizon in number of steps"""

ControlInput = np.ndarray
"""2xN, N is the horizon in number of steps"""

DeltaTimeVector = np.ndarray
"""1xN, N is the horizon in number of steps"""


class KinematicMotionModel:
    """
    Abstract base class for kinematic motion models.
    All kinematic motion models should inherit from this class and implement the `predict` method.
    """

    def __init__(self):

        self._jacobian = None  # Jacobian matrix for the kinematic model
        self.state_dim = 0  # Dimension of the state space
        self.input_dim = 0  # Dimension of the control input space
        self.name = "KinematicMotionModel"  # Name of the motion model
        self._jacobian_inv = None  # Inverse of the Jacobian matrix, if applicable
        self.control_input_frame = "body"  # "body", "joints"
        self.nb_group_state = 1  # Number of groups in the state, used for multi-group systems
        self._param_list = []

    # def compute_jacobian(self):
    #    """
    #    Define the Jacobian matrix for the kinematic motion model and its inverse
    #    This method should be implemented by subclasses to define the specific Jacobian.
    #    """
    #    raise NotImplementedError("Subclasses must implement this method to define the Jacobian matrix.")

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

        predicted_state = np.zeros((self.state_dim, dt.shape[1]))

        for i in range(body_commands.shape[1]):

            command = body_commands[0:, i]
            # Compute the change in state using the Jacobian
            delta_state = command * dt[0, i]

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
        N = dt.shape[1]

        initial_shape = (9, 1)
        if initial_state.shape != initial_shape:
            raise ValueError(f"inital_state shape should be {initial_shape}, not {initial_state.shape}")

        control_shape = (self.input_dim, N)
        if control_input.shape != control_shape:
            raise ValueError(f"control_input shape should be ({control_shape}), not {control_input.shape}")

        dt_shape = (1, N)
        if dt.shape != dt_shape:
            raise ValueError(f"dt shape should be ({dt_shape}), not {dt.shape}")

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
