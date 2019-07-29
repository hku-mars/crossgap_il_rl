import airsim
import numpy as np
from numpy import linalg as LA
import cv2 as cv2
import math
import time,sys
import dill as pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Torch version #####
sys.path.append("../")
from quad_rotor_controller_torch import Quad_rotor_controller
sys.path.append("../../deep_drone")
import trajectory_rnn_net

from tools.img_tools import img_tools
from tools.query_aimsim_images import Airsim_state_capture
from narrow_gap import narrow_gap
from narrow_gap import Quadrotor_painter
import transforms3d


import torch
from Rapid_trajectory_generator import Rapid_trajectory_generator

def generate_random_state():
    np.set_printoptions(precision= 2)
    for k in range(10):
        pos = np.random.random(size=( 3))*100
        spd = np.random.random(size=( 3))*100
        acc = np.random.random(size=( 3))*100
        print(pos, spd, acc)

if __name__ == "__main__":
    print("Hi~, this is trajectory generator.")
    data_size = 10000*100
    max_amp = 50.0
    logger_name = ("G:/data/narrow_gap/random/log_random_for_learn_%d.txt"%data_size )
    if_network = 0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    quad_painter = Quadrotor_painter(ax, 1)
    quad_rotor_controller = Quad_rotor_controller(logger_name, if_network=if_network)

    start_pos = [-0, 0, 10]
    quad_rotor_controller.init_airsim_client()
    quad_rotor_controller.refresh_state()

    quad_rotor_controller.quad_painter = quad_painter

    # quad_rotor_controller.move_to_pos(start_pos, if_enable_network=1)
    quad_rotor_controller.if_log = 1
    # while(1):
    # quad_rotor_controller.current
    for k in range(data_size):
        if( k % (data_size/10000)==0 ):
            print("k = ",k)
        amp = k*max_amp/data_size
        # self.set_target_body(pos, [0, 0, 0], [0, 0, 0], np.eye(3, 3))
        pos = np.random.random(size=(3)) * amp - 0.5 * amp
        spd = np.random.random(size=(3)) * amp - 0.5 * amp
        acc = np.random.random(size=(3)) * amp - 0.5 * amp
        quad_rotor_controller.set_target_body(pos.tolist(), spd.tolist(), acc.tolist(), np.eye(3,3))
        quad_rotor_controller.compute_target_rot(refreh_state=0)
        quad_rotor_controller.control(if_output = 0)
    quad_rotor_controller.save_log()
    quad_rotor_controller.logger.plot_data()
    # del quad_rotor_controller
    plt.show()