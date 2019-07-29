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

from narrow_gap import narrow_gap
from tf_policy_network import Policy_network
import Rapid_trajectory_generator
from Rapid_trajectory_generator import Rapid_trajectory_generator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU using
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def draw_curve(input_data,output_data, dl_data):
    LINE_SIZE = 8
    SIZE_BIAS = 32
    SIZE_DIFF = 6
    SMALL_SIZE = SIZE_BIAS - SIZE_DIFF
    MEDIUM_SIZE = SIZE_BIAS
    BIGGER_SIZE = SIZE_BIAS + SIZE_DIFF
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    # plt.rc('title', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure("comparison")
    plt.subplot(3, 1, 1)
    alias_tr = 'Tr'
    alias_dl = 'Dl'
    plt.plot(input_data[:, 0], output_data[:, 0], 'r--', label='%s_pos_x'%(alias_tr), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], output_data[:, 1], 'k--', label='%s_pos_y'%(alias_tr), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], output_data[:, 2], 'b--', label='%s_pos_z'%(alias_tr), linewidth=LINE_SIZE)

    plt.plot(input_data[:, 0], dl_data[:, 0], 'r-', label='%s_pos_x'%(alias_dl), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], dl_data[:, 1], 'k-', label='%s_pos_y'%(alias_dl), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], dl_data[:, 2], 'b-', label='%s_pos_z'%(alias_dl), linewidth=LINE_SIZE)
    plt.grid()
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 2)
    plt.plot(input_data[:, 0], output_data[:, 3], 'r--', label='%s_vel_x'%(alias_tr), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], output_data[:, 4], 'k--', label='%s_vel_y'%(alias_tr), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], output_data[:, 5], 'b--', label='%s_vel_z'%(alias_tr), linewidth=LINE_SIZE)

    plt.plot(input_data[:, 0], dl_data[:, 3], 'r-', label='%s_vel_x'%(alias_dl), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], dl_data[:, 4], 'k-', label='%s_vel_y'%(alias_dl), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], dl_data[:, 5], 'b-', label='%s_vel_z'%(alias_dl), linewidth=LINE_SIZE)
    plt.grid()
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 3)
    plt.plot(input_data[:, 0], output_data[:, 6], 'r--', label='%s_acc_x'%(alias_tr), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], output_data[:, 7], 'k--', label='%s_acc_y'%(alias_tr), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], output_data[:, 8], 'b--', label='%s_acc_z'%(alias_tr), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], dl_data[:, 6], 'r-', label='%s_acc_x'%(alias_dl), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], dl_data[:, 7], 'k-', label='%s_acc_y'%(alias_dl), linewidth=LINE_SIZE)
    plt.plot(input_data[:, 0], dl_data[:, 8], 'b-', label='%s_acc_z'%(alias_dl), linewidth=LINE_SIZE)
    plt.grid()
    plt.legend(loc='upper left')

    plt.pause(0.1)


def plot_data():
    import pickle
    pkl_name = 'comp_planning.pkl'
    offline_data = pickle.load(open(pkl_name, 'rb'))
    rapid_traj = Rapid_trajectory_generator()
    rapid_traj.input_data = offline_data['in']
    rapid_traj.output_data = offline_data['tr']
    tf_out = offline_data['dl']
    draw_curve(offline_data['in'], offline_data['tr'], offline_data['dl'])
    plt.show()
    # print(offline_data)

if __name__ == "__main__":
    print("Hello, this is test_policy_net.py")
    plot_data()
    exit(0)
    load_dir = "%s/planner" % (os.getenv('DL_data'))
    # traj_file_name = "%s/traj_cross.pkl"%load_dir
    traj_file_name = "%s/traj_0.pkl"%load_dir
    print("load test_file_name =  %s."%traj_file_name)
    policy_network = Policy_network()
    plan_net = policy_network.plan_network
    rapid_trajectory = Rapid_trajectory_generator()
    rapid_trajectory.load_from_file(traj_file_name)
    rapid_trajectory.plot_data("test")
    sess = policy_network.sess

    # start_pos = [1.0, 10, 4.5]
    if_compare = 1
    if(if_compare):
        start_pos = [-2 , -1, 0.5]
        plane_size = -0.0
        rad2angle = 180.0 / math.pi
        draw_traj = 0
        narrow_gap = narrow_gap([0 - plane_size, 0.5, 2.1], [0 / rad2angle, 0 / rad2angle, -60 / rad2angle]);
        # narrow_gap.para_cross_spd = 4.0
        narrow_gap.para_cross_spd = np.linalg.norm(np.array(narrow_gap.cross_path_p_0.T) - np.array(start_pos)) / 2.6

        narrow_gap.cross_ballistic_trajectory(if_draw=draw_traj)

        narrow_gap.approach_trajectory(start_pos,
                                       [0, 0, 0],
                                       [0, 0, 0], if_draw=0)

        learning_rate = 5e-6
        t_start = cv2.getTickCount()
        # rapid_trajectory.export_data(20, if_random= 0)
        # for loop in range(5):
        #     plan_net.train_step.run({plan_net.net_input: rapid_trajectory.input_data, plan_net.target_output: rapid_trajectory.output_data, plan_net.adam_learnning_rate: learning_rate})

        print('Cost time =', (cv2.getTickCount()-t_start)*1000.0/cv2.getTickFrequency() , ' ms')
    # tf_out_data = sess.run(plan_net.net_output, feed_dict={plan_net.net_input: rapid_trajectory.input_data})
    # rapid_trajectory.output_data = tf_out_data
    # rapid_trajectory.plot_data_dash("traditional")
    policy_network.compare_traditional_traj_and_network(narrow_gap)