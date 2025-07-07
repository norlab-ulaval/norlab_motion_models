from matplotlib import pyplot as plt
import numpy as np
from norlab_motion_models.ideal_diff_drive_2D import IdealDiffDrive2D, IdealDiffDrive2DParam

idd_param: IdealDiffDrive2DParam = {
    "wheel_radius": 0.1,
    "base_width": 2.0,
    "wheel_radius_gain": 1.0,
    "base_width_gain": 1.0,
}

idd = IdealDiffDrive2D(idd_param)


nb_steps = 20
init_state = np.zeros((9, 1))
command = np.ones((2, nb_steps))
dt = np.array([[1 / 20.0] * nb_steps])

predicted_states = idd.predict(init_state, command, dt)

plt.scatter(predicted_states[0, :], predicted_states[1, :])
plt.show()
