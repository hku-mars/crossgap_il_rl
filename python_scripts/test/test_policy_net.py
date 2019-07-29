##! /usr/local/bin/python3.5
import math
import numpy as np
import copy
import os
import json
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # Capactable with ros
except:
    pass
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


WORK_DIR = os.getenv("CROSS_GAP_WORK_DIR") # Getting path from system variable.
if WORK_DIR is None:
    WORK_DIR = "../"

sys.path.append("%s/cross_gap/"%WORK_DIR)
sys.path.append("%s/cross_gap/tools"%WORK_DIR)
sys.path.append("%s/deep_drone"%WORK_DIR)
sys.path.append("%s"%WORK_DIR)

save_dir = "%s/planner_testing" % (os.getenv('DL_data'))
from narrow_gap import narrow_gap
from tf_policy_network import Policy_network
from Rapid_trajectory_generator import Rapid_trajectory_generator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU using
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

if __name__ == "__main__":
    print("Hello, this is test_policy_net.py")
    load_dir = "../../data/planner"
    # traj_file_name = "%s/traj_cross.pkl"%load_dir
    traj_file_name = "%s/traj_0.pkl"%load_dir
    print("load test_file_name =  %s."%traj_file_name)
    policy_network = Policy_network()
    plan_net = policy_network.plan_network
    rapid_trajectory = Rapid_trajectory_generator()
    rapid_trajectory.load_from_file(traj_file_name)
    # rapid_trajectory.plot_data("test")
    sess = policy_network.sess

    # start_pos = [1.0, 10, 4.5]
    if_compare = 1
    if(if_compare):
        start_pos = [-5 , 0, 4.5]
        plane_size = -0.0
        rad2angle = 180.0 / math.pi
        draw_traj = 0
        narrow_gap = narrow_gap([10 - plane_size, 0, 5 - plane_size], [0 / rad2angle, 20 / rad2angle, -30 / rad2angle]);
        # narrow_gap.para_cross_spd = 4.0
        narrow_gap.para_cross_spd = np.linalg.norm(np.array(narrow_gap.cross_path_p_0.T) - np.array(start_pos)) / 5.0

        narrow_gap.cross_ballistic_trajectory(if_draw=draw_traj)

        narrow_gap.approach_trajectory(start_pos,
                                       [0, 0, 0],
                                       [0, 0, 0], if_draw=0)

        t_start = cv2.getTickCount()

        print('Cost time =', (cv2.getTickCount()-t_start)*1000.0/cv2.getTickFrequency() , ' ms')

    policy_network.compare_traditional_traj_and_network(narrow_gap)

    if plan_net.if_normalize:
        scale = sess.run(plan_net.scale_factor, feed_dict={plan_net.net_input: rapid_trajectory.input_data})
        input_normalize = sess.run(plan_net.input_normalize, feed_dict={plan_net.net_input: rapid_trajectory.input_data})

    tf_out_data = sess.run(plan_net.net_output, feed_dict={plan_net.net_input: rapid_trajectory.input_data})
    rapid_trajectory.output_data = tf_out_data
    rapid_trajectory.plot_data_dash("test")
    plt.show()
