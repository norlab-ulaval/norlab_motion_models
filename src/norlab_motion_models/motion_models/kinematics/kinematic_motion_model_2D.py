from ...motion_model import MotionModel
from scipy.spatial.transform import Rotation as R
import numpy as np

class KinematicMotionModel(MotionModel):
    """
    Abstract base class for kinematic motion models.
    All kinematic motion models should inherit from this class and implement the `predict` method.
    """

    def __init__(self):
        
        self._jacobian = None  # Jacobian matrix for the kinematic model
        self.state_dim = None  # Dimension of the state space
        self.control_dim = None  # Dimension of the control input space
        self.name = "KinematicMotionModel"  # Name of the motion model
        self._jacobian_inv = None  # Inverse of the Jacobian matrix, if applicable
        self.control_input_frame = "body" # "body", "joints"
        self.nb_group_state = 1 # Number of groups in the state, used for multi-group systems
    
    #def compute_jacobian(self):
    #    """
    #    Define the Jacobian matrix for the kinematic motion model and its inverse
    #    This method should be implemented by subclasses to define the specific Jacobian.
    #    """
    #    raise NotImplementedError("Subclasses must implement this method to define the Jacobian matrix.")
    
    def integrate_position(self, initial_state, speed_state, dt):
        """
        Integrate the position based on the current state, control input, and time step.
        Return the position in the initial state frame. 
        :param initial_state: Current state of the system.
        :param speed_state: speed_state (body velocity) to apply.
        :param dt: Time step for the integration.
        :return: Updated state after applying the control input.
        """
        
        if self._jacobian is None:
            raise ValueError("Jacobian matrix J is not initialized.")
        
        if initial_state.shape[0] != self.state_dim:
            raise ValueError("Initial state dimension does not match the state dimension of the model.")
        
        
        if speed_state.shape[0] != 6:
            raise ValueError("Control input dimension does not match the control dimension of the model.")

        #if dt.shape[0] != 1 or dt.shape[1] != self.speed_state.shape[1]:
        #    raise ValueError("Time step dt must be a vector with shape 1xcontrol_dim.")
        
    
        group_transform_i = []
        for group in range(self.nb_group_state):
            rotation= R.from_euler('z', initial_state[6*group+5, 0], degrees=False)
            transform = np.eye(4)
            transform[:3, :3] = rotation.as_matrix()
            transform[:3, 3] = initial_state[6*group:6*group+3, 0]
            #transform_i = np.eye(3)
            group_transform_i.append(transform)
        #transform_i = np.eye(3)
        # Compute the position in the initial state frame

        predicted_state = np.zeros((6*self.nb_group_state,dt.shape[0]))  
        
        for i in range(speed_state.shape[1]):
            
            for state_group in range(self.nb_group_state):
                transform_i = group_transform_i[state_group]
                group_speed_state = speed_state[state_group*6:6*(state_group+1),i]  #initial_state[3*state_group:3*(state_group+1), 0]
                # Compute the change in state using the Jacobian
                delta_state = group_speed_state * dt
                delta_lin = np.array([delta_state[0], delta_state[1],delta_state[2], 1]).T
                
                rotation= R.from_euler('z', delta_state[5], degrees=False)
                transform = np.eye(4)
                transform[:3, :3] = rotation.as_matrix()
                angles = rotation.as_euler('xyz', degrees=False)
                print(angles)
                predicted_state[6*state_group+3:6*(state_group+1),i] = angles
                predicted_state[6*state_group:6*(state_group+1)-3,i] = np.squeeze(transform_i @ delta_lin)[:3] # Add positions

                transform_i = transform_i @ transform
                theta = np.arctan2(transform_i[1, i], transform_i[0, i])
                
                group_transform_i[state_group] = transform_i
                predicted_state[6*state_group+5, i] = theta 

        return predicted_state

    def predict(self, initial_state, control_input, dt):
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
        if self.control_input_frame == "joints":
            ## Assuming the control input is in the joint_state
            speed_state = self.compute_fwd_kinematics( control_input) # 3xN command
        elif self.control_input_frame == "body":
            ## Assuming the control input is in the body frame
            speed_state = control_input
        else:
            raise ValueError("control_input_frame must be either 'joints' or 'body'.")
        
        print("twist_msg",speed_state)

        integrated_state = self.integrate_position(initial_state, speed_state, dt)

        return integrated_state

    def compute_fwd_kinematics(self, joint_state):
        """ Compute forward kinematics for a given joint state."""
        
        if self._jacobian is None:
            raise ValueError("Jacobian matrix J is not initialized.")
        if joint_state.shape[0] != self._jacobian.shape[1]:
            raise ValueError("Joint state dimension does not match the Jacobian matrix dimension.")

        return self._jacobian @ joint_state
    
    def compute_reverse_kinematics(self, end_effector_state):
        """ Compute reverse kinematics for a given end-effector state."""
        
        if self._jacobian is None:
            raise ValueError("Jacobian matrix J is not initialized.")
        if end_effector_state.shape[0] != self._jacobian.shape[0]:
            raise ValueError("End-effector state dimension does not match the Jacobian matrix dimension.")

        return self._jacobian.T @ end_effector_state
    

    def load_param(self, yaml_path):
        """
        Load the kinematic motion model parameters from a YAML file.
        """
        raise NotImplementedError()

    def safe_params(self):
        """
        Return a dictionary of safe parameters for the kinematic motion model.
        """
        raise NotImplementedError()