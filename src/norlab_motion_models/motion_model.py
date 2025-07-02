import abc
import numpy as np 
from typing import List, Dict


class MotionModel(abc.ABC):
    """
    Abstract base class for motion models.
    All motion models should inherit from this class and implement the `predict` method.
    """

    @abc.abstractmethod
    def load_params(self, yaml_path):
        """
        Load the motion model parameters from a YAML file.

        :param yaml_path: Path to the YAML file containing model parameters.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def save_params(self):
        """
        Return a dictionary of safe parameters for the motion model.
        Safe parameters are those that can be used without risk of causing errors or unexpected behavior.
        
        :return: Dictionary of safe parameters.
        """
        raise NotImplementedError()
    
    #@abc.abstractmethod
    #def predict(self, state, control_input, dt):
    #    """
    #    Predict the next state based on the current state, control input, and time step.

    #    :param state: Current state of the system.
    #    :param control_input: Control input to apply.
    #    :param dt: Time step for the prediction.
    #    :return: Predicted next state.
    #    """
    #    raise NotImplementedError()

    #@abc.abstractmethod
    #def predict_horizon(self, state, control_inputs, dt):
    #    """
    #    Predict the state over a horizon given a sequence of control inputs.

    #    :param state: Initial state of the system.
    #    :param control_inputs: List of control inputs to apply at each time step.
    #    :param dt: Time step for the prediction.
    #    :return: List of predicted states over the horizon.
    #    """
    #    predicted_states = [state]
    #    for control_input in control_inputs:
    #        state = self.predict(state, control_input, dt)
    #        predicted_states.append(state)
    #    
    #    raise NotImplementedError()
    
    #@abc.abstractmethod
    #def evaluate_prediction_time(self, state, control_input, dt):
    #    """
    #    Evaluate the time taken to predict the next state.

    #    :param state: Current state of the system.
    #    :param control_input: Control input to apply.
    #    :param dt: Time step for the prediction.
    #    :return: Time taken to perform the prediction.
    #    """
    #    raise NotImplementedError()
    
