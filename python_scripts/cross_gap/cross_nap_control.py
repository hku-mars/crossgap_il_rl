import airsim
import numpy as np
from numpy import linalg as LA
import cv2 as cv2
import math
import time,sys
import dill as pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from Rapid_trajectory_generator import Rapid_trajectory_generator
if_network = 0

# Torch version #####
if(if_network == 0):
    from quad_rotor_controller_torch import Quad_rotor_controller
    sys.path.append("../deep_drone")
    import trajectory_rnn_net
else:
    ##### Tf no gym version #####
    # from quad_rotor_controller_tf import Quad_rotor_controller
    ##### Tf gym version #####
    from quad_rotor_controller_tf_gym import Quad_rotor_controller

from tools.img_tools import img_tools
from tools.query_aimsim_images import Airsim_state_capture
from narrow_gap import narrow_gap
from narrow_gap import Quadrotor_painter
import transforms3d

import torch
from Rapid_trajectory_generator import Rapid_trajectory_generator

if __name__ == "__main__":
    fig = plt.figure()
    if_replan = 0
    if_planner_network = 1
    if_connect_to_simulator = 1
    rad2angle = 180.0 / math.pi
    # fig = plt.figure()
    plt.close('all')
    plt.tight_layout()
    ax = fig.gca(projection='3d')
    # start_pos = [-0, 0, 5]
    # start_pos = [-10, 10, 10]
    # start_pos = [ -10 , 10, 4.5]
    start_pos = [0, 0, 0]
    plane_size = 0.0
    narrow_gap = narrow_gap([2 - plane_size, 1.0, 1.5], [0 / rad2angle, 0 / rad2angle, 60 / rad2angle]);
    # narrow_gap = narrow_gap([10 - plane_size, 0, 5-plane_size], [0 / rad2angle, 0 / rad2angle, -30 / rad2angle]);
    # narrow_gap.para_cross_spd = 6.0
    narrow_gap.para_cross_spd = np.linalg.norm(np.array(narrow_gap.cross_path_p_0.T) - np.array(start_pos)) /2.7

    logger_name = ("%s/narrow_gap/log_%d_%d_%d_%.1f_no_learn.txt" % (os.getenv("DL_data"), start_pos[0], start_pos[1], start_pos[2], narrow_gap.para_cross_spd))
    draw_traj = 0
    if draw_traj:
        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(0, 15)
        ax.set_ylim(-7.5, 7.5)
        ax.set_zlim(0, 15)
        narrow_gap.set_plot_figure(ax)
        narrow_gap.draw_rotate_rectangle()

    # compare(narrow_gap)
    # exit()
    if draw_traj:
        plt.show()

    quad_painter = Quadrotor_painter(ax, 1)
    quad_rotor_controller = Quad_rotor_controller(logger_name, if_network=if_network)


    if if_connect_to_simulator:

        quad_rotor_controller.init_airsim_client()
        quad_rotor_controller.quad_painter = quad_painter
        quad_rotor_controller.if_log = 0
        quad_rotor_controller.move_to_pos(start_pos, if_enable_network = if_network)
        # exit()
        quad_rotor_controller.target_roll_vec_preference = np.matrix([1, 0, 0]).T
        # quad_rotor_controller.set_target_body([10, 10, 10], [0, 0, 0], [0, 0, 0], [0, 0, 0])

        quad_rotor_controller.refresh_state()
        quad_rotor_controller.if_log = 1
        narrow_gap.approach_trajectory(quad_rotor_controller.current_pos,
                                       quad_rotor_controller.current_spd,
                                       quad_rotor_controller.current_acc, if_draw=draw_traj)
        narrow_gap.approach_trajectory(quad_rotor_controller.current_pos,
                                       quad_rotor_controller.current_spd,
                                       quad_rotor_controller.current_acc, if_draw=draw_traj)
        print("Tf = %.2f, t_c = %.2f" % (narrow_gap.Tf, narrow_gap.t_c))
        # print(narrow_gap.get_rotation(3.5))
        t_start = cv2.getTickCount()
        while True:
            t = (cv2.getTickCount() - t_start) / cv2.getTickFrequency()
            narrow_gap.rapid_nn_input_vector[0] = t
            if (t <= narrow_gap.Tf + narrow_gap.t_c * 2):
                target_pos = narrow_gap.get_position(t)
                target_vel = narrow_gap.get_velocity(t)
                target_acc = narrow_gap.get_acceleration(t)

                if (t < narrow_gap.Tf- narrow_gap.t_c ):
                    if (if_network):
                        if (if_planner_network):
                            quad_rotor_controller.hidden_state = None
                            quad_rotor_controller.follow_rapid_trajectory(narrow_gap.rapid_nn_input_vector, narrow_gap.rapid_pos_start)
                        else:
                            target_pos = narrow_gap.get_position(t)
                            target_vel = narrow_gap.get_velocity(t)
                            target_acc = narrow_gap.get_acceleration(t)

                            quad_rotor_controller.set_target_body(target_pos, target_vel, target_acc, np.eye(3, 3))
                            quad_rotor_controller.control_using_network()
                    else:
                        quad_rotor_controller.set_target_body(target_pos, target_vel, target_acc, np.eye(3, 3))
                        quad_rotor_controller.compute_target_rot()
                        quad_rotor_controller.control()
                else:
                    quad_rotor_controller.set_target_body(target_pos, target_vel, target_acc, np.eye(3, 3))
                    quad_rotor_controller.compute_target_rot()
                    quad_rotor_controller.control()
            else:
                quad_rotor_controller.set_target_body(target_pos, [0, 0, 0], [0, 0, 0], np.eye(3, 3))
                # quad_rotor_controller.client.hoverAsync()
                quad_rotor_controller.hover()
                quad_rotor_controller.save_log()
                break
            # time.sleep(0.04)
        print("exit")
        quad_rotor_controller.logger.plot_data()
        del quad_rotor_controller
        plt.show()