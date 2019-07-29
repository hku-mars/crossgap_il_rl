import numpy as np
import airsim
import matplotlib.pyplot as plt
import math
import transforms3d
from transforms3d import *
from transforms3d.euler import TBZYX, EulerFuncs

# <MultirotorState> {   'collision': <CollisionInfo> {   'has_collided': False,
#     'impact_point': <Vector3r> {   'x_val': 0.0,
#     'y_val': 0.0,
#     'z_val': 0.0},
#     'normal': <Vector3r> {   'x_val': 0.0,
#     'y_val': 0.0,
#     'z_val': 0.0},
#     'object_id': -1,
#     'object_name': '',
#     'penetration_depth': 0.0,
#     'position': <Vector3r> {   'x_val': 0.0,
#     'y_val': 0.0,
#     'z_val': 0.0},
#     'time_stamp': 0},
#     'gps_location': <GeoPoint> {   'altitude': 122.14649963378906,
#     'latitude': 47.641468,
#     'longitude': -122.140165},
#     'kinematics_estimated': <KinematicsState> {   'angular_acceleration': <Vector3r> {   'x_val': -3.84734903491335e-06,
#     'y_val': 3.7309626350179315e-05,
#     'z_val': 4.788217211171286e-06},
#     'angular_velocity': <Vector3r> {   'x_val': -0.04477997496724129,
#     'y_val': 0.0974305272102356,
#     'z_val': 0.26188379526138306},
#     'linear_acceleration': <Vector3r> {   'x_val': -3.0163817405700684,
#     'y_val': -0.038107872009277344,
#     'z_val': 0.6393585205078125},
#     'linear_velocity': <Vector3r> {   'x_val': -10.04089069366455,
#     'y_val': 7.450755596160889,
#     'z_val': -3.3728837966918945},
#     'orientation': <Quaternionr> {   'w_val': 0.8526145219802856,
#     'x_val': 0.1125524714589119,
#     'y_val': 0.1568954586982727,
#     'z_val': 0.48555561900138855},
#     'position': <Vector3r> {   'x_val': -17.527751922607422,
#     'y_val': 18.932405471801758,
#     'z_val': -50.52725601196289}},
#     'landed_state': 1,
#     'rc_data': <RCData> {   'is_initialized': False,
#     'is_valid': True,
#     'left_z': 0.0,
#     'pitch': -0.0,
#     'right_z': 0.0,
#     'roll': 0.0,
#     'switches': 0,
#     'throttle': 0.5,
#     'timestamp': 0,
#     'vendor_id': 'VID_045E',
#     'yaw': 0.0},
#     'timestamp': 1532674989774579712}

def plot_state_vector_linear(state_vector):
    pos_x = []
    pos_y = []
    pos_z = []

    vel_x = []
    vel_y = []
    vel_z = []

    acc_x = []
    acc_y = []
    acc_z = []
    for state in state_vector:

        pos_x.append(state.kinematics_estimated.position.x_val)
        pos_y.append(state.kinematics_estimated.position.y_val)
        pos_z.append(state.kinematics_estimated.position.z_val)

        vel_x.append(state.kinematics_estimated.linear_velocity.x_val)
        vel_y.append(state.kinematics_estimated.linear_velocity.y_val)
        vel_z.append(state.kinematics_estimated.linear_velocity.z_val)

        acc_x.append(state.kinematics_estimated.linear_acceleration.x_val)
        acc_y.append(state.kinematics_estimated.linear_acceleration.y_val)
        acc_z.append(state.kinematics_estimated.linear_acceleration.z_val)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(pos_x, 'r', label="r", marker="o")
    plt.plot(pos_y, 'g', label="p", marker="v")
    plt.plot(pos_z, 'b', label="y", marker="^")
    plt.title("angular_pos")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(vel_x, 'r', label="x", marker="o")
    plt.plot(vel_y, 'g', label="y", marker="v")
    plt.plot(vel_z, 'b', label="z", marker="^")
    plt.title("angular_vel")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(acc_x, 'r', label="x", marker="o")
    plt.plot(acc_y, 'g', label="y", marker="v")
    plt.plot(acc_z, 'b', label="z", marker="^")
    plt.title("angular_acc")
    plt.legend()
    plt.suptitle("linear")
    plt.pause(1)

def plot_state_vector_angular(state_vector):
    pos_x = []
    pos_y = []
    pos_z = []

    vel_x = []
    vel_y = []
    vel_z = []

    acc_x = []
    acc_y = []
    acc_z = []
    for state in state_vector:
        ori_q = state.kinematics_estimated.orientation
        q = [ori_q.w_val, ori_q.x_val,ori_q.y_val,ori_q.z_val]
        r, p, y = transforms3d.euler.quat2euler(q, axes='sxyz')
        pos_x.append(r*180.0/math.pi)
        pos_y.append(p*180.0/math.pi)
        pos_z.append(y*180.0/math.pi)

        # pos_x.append(raw_state.kinematics_estimated.orientation.x_val)
        # pos_y.append(raw_state.kinematics_estimated.orientation.y_val)
        # pos_z.append(raw_state.kinematics_estimated.orientation.z_val)

        vel_x.append(state.kinematics_estimated.angular_velocity.x_val)
        vel_y.append(state.kinematics_estimated.angular_velocity.y_val)
        vel_z.append(state.kinematics_estimated.angular_velocity.z_val)

        acc_x.append(state.kinematics_estimated.angular_acceleration.x_val)
        acc_y.append(state.kinematics_estimated.angular_acceleration.y_val)
        acc_z.append(state.kinematics_estimated.angular_acceleration.z_val)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(pos_x, 'r', label="r",marker = "o")
    plt.plot(pos_y, 'g', label="p",marker = "v")
    plt.plot(pos_z, 'b', label="y",marker = "^")
    plt.title("angular")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(vel_x, 'r', label="r",marker = "o")
    plt.plot(vel_y, 'g', label="p",marker = "v")
    plt.plot(vel_z, 'b', label="y",marker = "^")
    plt.title("angular_vel")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(acc_x, 'r', label="r",marker = "o")
    plt.plot(acc_y, 'g', label="p",marker = "v")
    plt.plot(acc_z, 'b', label="y",marker = "^")
    plt.title("angular_acc")
    plt.suptitle("Angular")
    plt.legend()
    plt.pause(1)