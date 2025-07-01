import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from norlab_motion_models.motion_models.kinematics.ideal_diff_drive_2D import IdealDiffDrive2D 
import yaml

LENGTH = 2
axis_lim = 10
XLIM = (-axis_lim, axis_lim)
YLIM = (-axis_lim, axis_lim)
slider_bottom = 0.05
slider_spacing = 0.05
slider_height = 0.03    
slider_width = 0.5
fig_height = 10
fig_width = 10

# Load parameters from YAML file

with open("motion_model_vizualization/IDD_2D_params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Initialize the robot model with parameters from YAML
robot = IdealDiffDrive2D(params["robot_params"])
# Initial state and command
x_init = np.array([[0.0, 0.0, 0.0,0.0,0.0,0.0]*robot.nb_group_state]).T
u_init = np.zeros(robot.input_dim)  # Use robot's input dimension
x_current = x_init.copy()
trajectory = x_current.copy()
sim_params = params["simulation_params"]
dt = sim_params["dt"]
print("IdealDiffDrive2D parameters:", params["robot_params"])
# -----------------------------------------------
# Set up plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
fig.set_size_inches(fig_width, fig_height)

trajectory_line_group = []
trajectory_point_group = []
for i in range(robot.nb_group_state):
    trajectory_line, = ax.plot([], [], '-', label=f'trajectory group {i+1}')
    trajectory_points, = ax.plot([], [], 'o', markersize=5, label=f'positions group {i+1}')
    trajectory_line_group.append(trajectory_line)
    trajectory_point_group.append(trajectory_points)


import matplotlib.patches as patches

def draw_frame(ax, x, length=LENGTH):
    if hasattr(draw_frame, 'quivers'):
        for q in draw_frame.quivers:
            q.remove()
    if hasattr(draw_frame, 'rect'):
        draw_frame.rect.remove()
    origin = x[:2]
    theta = x[5]
    print(x)
    x_axis = length * np.array([np.cos(theta), np.sin(theta)])
    y_axis = length * np.array([-np.sin(theta), np.cos(theta)])
    qx = ax.quiver(origin[0],origin[1], *x_axis, angles='xy', scale_units='xy', scale=1, color='r', width=0.01)
    qy = ax.quiver(origin[0],origin[1], *y_axis,angles='xy', scale_units='xy', scale=1, color='g', width=0.01)
    draw_frame.quivers = [qx, qy]
    # Draw rectangle representing the robot
    rect_length = length
    rect_width = length * 0.5
    # Rectangle is centered at origin, so shift to center
    rect = patches.Rectangle(
        (origin[0] - rect_length/2, origin[1] - rect_width/2),
        rect_length, rect_width,
        rotation_point="center",
        angle=np.degrees(theta),
        linewidth=1, edgecolor='b', facecolor='none', zorder=3
    )
    ax.add_patch(rect)
    draw_frame.rect = rect

ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal', adjustable='box')
ax.set_title('Interactive Robot Motion')
ax.legend()

# Dynamically create sliders for each input
axcolor = 'lightgoldenrodyellow'
slider_axes = []
sliders = []

for i in range(robot.input_dim):

    rect = [0.5-slider_width/2, slider_bottom + i * slider_spacing, slider_width, slider_height]
    ax_slider = plt.axes(rect, facecolor=axcolor)
    slider_axes.append(ax_slider)
    label = sim_params["input_labels"][i]
    limits = sim_params["input_limits"][i]
    slider = Slider(ax_slider, label, limits[0], limits[1], valinit=sim_params["u_init"][i])
    sliders.append(slider)

def on_key(event):
    global x_current, trajectory
    if event.key == ' ':
        u = np.flip(np.array([[slider.val for slider in sliders]]).T)
        dt = np.ones(1) * sim_params["dt"]
        print(x_current)
        x_current[:] = robot.predict(x_current, u,dt)
        print("x_current",x_current)
        trajectory=np.hstack((trajectory,x_current.copy()))
        traj_arr = trajectory
        for i in range(robot.nb_group_state):
            trajectory_line = trajectory_line_group[i]
            trajectory_points = trajectory_point_group[i]
            traj_arr = trajectory[6*i:6*i+2, :]
            #
        trajectory_line.set_data(traj_arr[0,:], traj_arr[1,:])
        trajectory_points.set_data(traj_arr[0, :], traj_arr[1,:])
        
        draw_frame(ax, x_current)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)

traj_arr = np.array(trajectory)
print(traj_arr)
for i in range(robot.nb_group_state):
    trajectory_line = trajectory_line_group[i]
    trajectory_points = trajectory_point_group[i]
    trajectory_line.set_data(traj_arr[6*i, :], traj_arr[6*i+1, :])
    trajectory_points.set_data(traj_arr[6*i, :], traj_arr[6*i+1, :])

draw_frame(ax, x_current)

plt.show()
