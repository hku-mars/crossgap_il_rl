import numpy as np
import cv2
import transforms3d
import airsim
import math
from numpy import linalg as LA
from tools.plane_logger import plane_logger

import torch
import torch.nn
import sys, os

sys.path.append("..")  # Adds higher directory to python modules path.
sys.path.append("../tools")  # Adds higher directory to python modules path.
from deep_drone.trajectory_rnn_net import Rnn_net
from deep_drone.trajectory_rnn_net import Rnn_net as trajectory_rnn_net
from pid_controller import Pid_controller
POS_ERR_MAX = 1.0
VEL_ERR_MAX = 10.0

class Quad_rotor_controller():
    def __init__(self, logger_name="plane_log.txt", if_network=0):
        self.if_debug = 0
        self.if_log = 1
        self.if_save_log = 1
        self.if_rnn_network = 0
        self.logger = plane_logger(logger_name)

        self.client = airsim.MultirotorClient()
        self.current_pos = np.zeros([3, 1])
        self.current_spd = np.zeros([3, 1])
        self.current_acc = np.zeros([3, 1])
        self.current_rot = np.eye(3, 3)

        self.target_pos = np.zeros([3, 1])
        self.target_spd = np.zeros([3, 1])
        self.target_acc = np.zeros([3, 1])
        self.target_rot = np.eye(3, 3)
        print(self.target_rot)

        self.target_force = np.zeros([3, 1])
        self.target_force_max = 5
        self.target_roll_vec_preference = np.matrix([1, 0, 0]).T
        self.target_roll_vec = np.matrix([1, 0, 0]).T
        self.target_pitch_vec = np.matrix([0, 1, 0]).T
        self.target_yaw_vec = np.matrix([0, 0, 1]).T
        self.pid_ctrl = Pid_controller()

        self.set_hover_pid_parameter()

        self.transform_R_world_to_ned = np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).T
        self.control_amp = 1
        # self.Kp = self.control_amp * 20 * np.eye(3, 3)
        # self.Kv = self.control_amp * 20 * np.eye(3, 3)

        self.Kp = 2.0 * np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 3]])
        self.Kv = self.control_amp * 10.0 * np.eye(3, 3)

        self.Ka = 0.95 * np.eye(3, 3)
        self.target_force_max = 2000



        self.gravity = np.array([[0], [0], [-9.8]])
        self.quad_painter = []
        self.t_start = cv2.getTickCount()
        self.if_network = if_network
        if (self.if_network):
            self.if_save_log = 0
            if (self.if_rnn_network):
                print("Load rnn net")
                self.control_network = Rnn_net()
                self.hidden_state = None
                self.control_network = torch.load("../deep_drone/demo_network_final_rnn.pkl").cpu()
            else:
                # self.control_network = torch.load("../deep_drone/demo_network_final.pkl").cpu()
                self.control_network = torch.load("../deep_drone/demo_network_50000.pkl").cpu()
            self.plan_network = torch.load("../deep_drone/rapid_traj_network_7362000.pkl").cpu()
            print(self.control_network)
            self.nn_input_vec = np.zeros([1, 9])
            self.nn_output_vec = np.zeros([1, 4])
            self.torch_input_vec = torch.from_numpy(self.nn_input_vec).float()
            self.torch_output_vec = torch.from_numpy(self.nn_output_vec).float()
            print(self.control_network)
        print("Init finish")

    def set_hover_pid_parameter(self):

        Kp = [4.5, 4.5, 4.5]
        Kv = [2.25, 2.25, 4.50]
        Ka = [0.9, 0.9, 0.9]

        self.pid_ctrl.set_parameter(Kp, Kv, Ka, g=9.8, mass=0.90)

    def set_tracking_pid_parameter(self):

        Kp = [6.5, 6.5, 6.5]
        Kv = [2.25, 2.25, 4.50]
        Ka = [0.9, 0.9, 0.9]

        self.pid_ctrl.set_parameter(Kp, Kv, Ka, g=9.8, mass=0.90)

    def save_log(self):
        if (self.if_save_log):
            self.logger.save()

    def init_airsim_client(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.reset()
        self.client.enableApiControl(True)

    def reset_client(self):
        self.client.reset()

    def set_target_body(self, target_pos, target_spd, target_acc, target_rot):
        self.target_pos = np.matrix(target_pos).T
        self.target_spd = np.matrix(target_spd).T
        self.target_acc = np.matrix(target_acc).T
        # self.target_rot = np.matrix(target_rot).T

    def set_target_world(self, pos_world, spd_world, acc_world, rot_world):
        self.target_pos = self.world_to_ned_coor(np.matrix(pos_world).T)
        self.target_spd = self.world_to_ned_coor(np.matrix(spd_world).T)
        self.target_acc = self.world_to_ned_coor(np.matrix(acc_world).T)

    def move_to_pos(self, pos, if_enable_network=1):
        # self.if_log = 0
        print("Move to pos:", pos)
        self.set_hover_pid_parameter()
        self.set_target_body(pos, [0, 0, 0], [0, 0, 0], np.eye(3, 3))
        max_pos_err = 1.0
        max_spd = 0.5
        max_acc = 0.5
        while (1):
            self.compute_target_rot()
            if (self.if_network and if_enable_network and self.if_rnn_network == 1):
                # self.control()
                self.control_using_network()
            else:
                self.control()
            # self.control_using_network()
            pos_err = self.target_pos - self.current_pos
            spd_err = self.target_spd - self.current_spd
            acc_err = self.target_acc - self.current_acc
            print(self.current_pos.ravel())
            if (LA.norm(pos_err) < max_pos_err):
                # print(pos_err)
                if (LA.norm(spd_err) < max_spd):
                    if (LA.norm(acc_err) < max_acc):
                        self.set_tracking_pid_parameter()
                        break

        print("Move to pos:", pos, "finish")
        # self.client.moveByAngleThrottleAsync(0,0,0,0,3e8)
        # time.sleep(0.1)

    def move_to_pos_airsim_api(self, pos):
        pos_ned = self.world_to_ned_coor(np.matrix(pos).T).T.tolist()[0]
        print("Move to ", pos_ned)
        self.client.moveToPositionAsync(pos_ned[0], pos_ned[1], pos_ned[2], 2, 3e8).join()
        self.client.moveToPositionAsync(pos_ned[0], pos_ned[1], pos_ned[2], 2, 3e8).join()
        # self.client.moveByAngleThrottleAsync(0, 0, 1, 0, 0.1)
        self.client.hoverAsync().join()
        while (1):
            time.sleep(0.01)
            state = self.client.getMultirotorState()
            max_hover_speed = 0.1
            if (state.kinematics_estimated.linear_velocity.x_val >= max_hover_speed or
                    state.kinematics_estimated.linear_velocity.y_val >= max_hover_speed or
                    state.kinematics_estimated.linear_velocity.z_val >= max_hover_speed):
                self.client.hoverAsync().join()
            else:
                state = self.client.getMultirotorState()
                state = self.client.getMultirotorState()
                self.refresh_state()
                print("current_pos = ", self.current_pos)
                break

    def compute_force(self):
        pass

    def compute_paw(self):
        pass

    def cos_theta_g_e(self, g, _e):
        e = _e / (LA.norm(_e) + 0.00000001)
        cos_theta = float((g.T * e) / (LA.norm(g) * LA.norm(e) + 0.00000001))
        return cos_theta

    def vector_project(self, e1, e2):  # e1 project on e2, e1 and e2 most be a vector (shape = [x,1])
        e2 = e2 / (LA.norm(e2) + 0.00000001)
        cos_theta = self.cos_theta_g_e(e1, e2)
        g_proj = LA.norm(e1) * cos_theta * e2
        return g_proj

    def compute_target_rot(self,refreh_state = 1):
        self.target_rot = np.eye(3, 3)
        if(refreh_state):
            self.refresh_state()
        pos_err = self.target_pos - self.current_pos
        vel_err = self.target_spd - self.current_spd
        acc_err = self.target_acc - self.current_acc

        if (LA.norm(pos_err) >POS_ERR_MAX):
            pos_err = pos_err*POS_ERR_MAX / LA.norm(pos_err)

        if (LA.norm(vel_err) > VEL_ERR_MAX):
            vel_err = vel_err*VEL_ERR_MAX/ LA.norm(vel_err)

        err_pos = self.Kp * pos_err
        err_vel = self.Kv* ( vel_err + err_pos)

        # self.target_force = -self.Kp * pos_err - self.Kv * vel_err + self.gravity + self.target_acc
        self.target_force =  ( err_vel + self.Ka * self.target_acc)


        self.target_force = (self.target_force - self.gravity)

        if (LA.norm(self.target_force) > self.target_force_max):
            self.target_force = self.target_force * self.target_force_max / LA.norm(self.target_force)

        if(self.target_force[2] <= 0.001):
            self.target_force[2] = 0.001

        # print((self.target_pos - self.current_pos).T , " --- ", self.target_force.T)
            # print("Force = ", self.target_force.T)
        if (self.if_debug):
            print("Force = ", self.target_force.T)
        self.target_yaw_vec = self.target_force / LA.norm(self.target_force)
        if (self.target_force[2] > 0):
            self.target_yaw_vec = self.target_yaw_vec * -1

        # TODO
        self.target_roll_vec = self.target_roll_vec_preference - self.vector_project(self.target_roll_vec_preference, self.target_yaw_vec);
        self.target_roll_vec = self.target_roll_vec / LA.norm(self.target_roll_vec)

        self.target_pitch_vec = np.cross(self.target_yaw_vec.T, self.target_roll_vec.T).T
        self.target_pitch_vec = self.target_pitch_vec / LA.norm(self.target_pitch_vec)

        # print("Yaw = ", self.target_yaw_vec.T)
        # print("Roll = ", self.target_roll_vec.T)
        # print("Pitch = ", self.target_pitch_vec.T)
        # print("Norm = ", LA.norm(self.target_yaw_vec), LA.norm(self.target_roll_vec), LA.norm(self.target_pitch_vec))

        self.target_rot[:, 0] = self.target_roll_vec.T.tolist()[0]
        self.target_rot[:, 1] = self.target_pitch_vec.T.tolist()[0]
        self.target_rot[:, 2] = self.target_yaw_vec.T.tolist()[0]
        # print(self.target_rot)

    def world_to_ned_coor(self, vector):
        return self.transform_R_world_to_ned * vector

    def ned_to_world(self, vector):
        return self.transform_R_world_to_ned.T * vector

    def refresh_state(self):
        state = self.client.getMultirotorState()
        self.raw_state = state
        self.current_pos = self.ned_to_world(np.matrix([state.kinematics_estimated.position.x_val,
                                                        state.kinematics_estimated.position.y_val,
                                                        state.kinematics_estimated.position.z_val]).T)

        self.current_spd = self.ned_to_world(np.matrix([state.kinematics_estimated.linear_velocity.x_val,
                                                        state.kinematics_estimated.linear_velocity.y_val,
                                                        state.kinematics_estimated.linear_velocity.z_val]).T)

        self.current_acc = self.ned_to_world(np.matrix([state.kinematics_estimated.linear_acceleration.x_val,
                                                        state.kinematics_estimated.linear_acceleration.y_val,
                                                        state.kinematics_estimated.linear_acceleration.z_val]).T)

        ori_q = state.kinematics_estimated.orientation
        q = [ori_q.w_val, ori_q.x_val, ori_q.y_val, ori_q.z_val]
        self.current_rot = self.transform_R_world_to_ned.T * transforms3d.euler.quat2mat(q)
        #
        # self.pid_ctrl.update_state( self.current_pos.ravel().tolist()[0],
        #                             self.current_spd.ravel().tolist()[0],
        #                               self.current_acc.ravel().tolist()[0] , q)
        self.pid_ctrl.update_state( self.current_pos.ravel().tolist()[0],
                                    self.current_spd.ravel().tolist()[0],
                                    [0,0,0], q)

        if (self.if_debug):
            print("---------------")
            print("Current_pos = ", self.current_pos.T)
            # print("Current_pos = ", self.target_pos.T)
            print("Current_spd = ", self.current_spd.T)
            # print("Current_pos = ", self.target_spd.T)

    def compute_yaw_rate(self):
        cur_roll_vec = self.current_rot[:, 0]
        tar_roll_vec = self.target_rot[:, 0]
        current_yaw = math.atan2(cur_roll_vec[1], cur_roll_vec[0])
        target_yaw = math.atan2(tar_roll_vec[1], tar_roll_vec[0])
        pass
        return -self.control_amp * 1 * (2 * (target_yaw - current_yaw) - self.control_amp * 0.1 * (-1 * self.raw_state.kinematics_estimated.angular_acceleration.z_val))

    def pid_rpy_to_air_sim_rpy(self, res_r, res_p, res_y = 0):

        mat_new_mat = transforms3d.euler.euler2quat(res_y, res_p, res_r, axes='sxyz')
        res_r, res_p, res_y = transforms3d.euler.quat2euler(mat_new_mat, axes='szyx')

        return res_r, -res_p, res_y

    def control(self,if_output = 1):
        self.current_time = (cv2.getTickCount() - self.t_start) / cv2.getTickFrequency()
        self.pid_ctrl.update_target( self.target_pos.ravel().tolist()[0]  ,self.target_spd.ravel().tolist()[0]  ,self.target_acc.ravel().tolist()[0]  , target_yaw = 0 )
        self.pid_ctrl.update_control()

        # r, p, y = transforms3d.euler.mat2euler(np.matmul(self.transform_R_world_to_ned, self.pid_ctrl.tar_r_mat ), axes='sxyz')
        r, p, y = transforms3d.euler.mat2euler(( self.pid_ctrl.tar_r_mat ), axes='sxyz')
        r = -r
        res_r,res_p, res_y = self.pid_rpy_to_air_sim_rpy(self.pid_ctrl.tar_roll, self.pid_ctrl.tar_pitch, self.pid_ctrl.tar_yaw)

        # print("Target_roll_pitch = [%.3f, %.3f, %.3f] "%(res_r * 57.3, res_p*57.3, res_y*57.3),  " using_roll_pitch = [%.3f, %.3f, %.3f]"%(r * 57.3, p*57.3, y*57.3))
        self.target_rot = np.matmul(self.transform_R_world_to_ned, self.pid_ctrl.tar_r_mat)
        r = res_r
        p = res_p
        # p = self.pid_ctrl.tar_pitch
        # y = self.pid_ctrl.tar_yaw

        # throttle = LA.norm(self.target_force) / 9.8 - (1 - 0.60)
        throttle = (self.pid_ctrl.tar_thrust ) / 9.8

        yaw_rate = self.compute_yaw_rate()
        # throttle = LA.norm(self.target_force) / 9.8 - 0.4
        if (self.if_debug):
            print( r * 57.3, p * 57.3, y * 57.3, throttle, yaw_rate)
        max_throttle = 10.0
        if (throttle < -max_throttle):
            throttle = -max_throttle
        if (throttle > max_throttle):
            throttle = max_throttle

        if(if_output):
            # print("[Out]: ", r * 57.3, p * 57.3, y * 57.3, throttle, yaw_rate)
            self.client.moveByAngleThrottleAsync(p, r, throttle, 0, 3e8)
        # plt.show()
        if (self.if_log):
            self.logger.update("time_stamp", self.current_time)
            self.logger.update("current_pos", self.current_pos)
            self.logger.update("current_spd", self.current_spd)
            self.logger.update("current_acc", self.current_acc)
            self.logger.update("current_rot", self.current_rot)

            self.logger.update("target_pos", self.target_pos)
            self.logger.update("target_spd", self.target_spd)
            self.logger.update("target_acc", self.target_acc)
            self.logger.update("target_rot", self.target_rot)

            self.logger.update("input_roll", r)
            self.logger.update("input_pitch", p)
            self.logger.update("input_yaw", y)
            self.logger.update("input_throttle", throttle)

    def hover(self):
        self.client.hoverAsync()

    def contorl_attitude(self, rot):
        self.target_rot = rot
        self.control()

    def control_using_network(self):
        self.refresh_state()
        self.current_time = (cv2.getTickCount() - self.t_start) / cv2.getTickFrequency()
        pos_err_3 = self.target_pos - self.current_pos
        spd_err_3 = self.target_spd - self.current_spd
        acc_err_3 = self.target_acc - self.current_acc

        self.nn_input_vec[0, 0] = float(pos_err_3[0])
        self.nn_input_vec[0, 1] = float(pos_err_3[1])
        self.nn_input_vec[0, 2] = float(pos_err_3[2])
        self.nn_input_vec[0, 3] = float(spd_err_3[0])
        self.nn_input_vec[0, 4] = float(spd_err_3[1])
        self.nn_input_vec[0, 5] = float(spd_err_3[2])
        self.nn_input_vec[0, 6] = float(acc_err_3[0])
        self.nn_input_vec[0, 7] = float(acc_err_3[1])
        self.nn_input_vec[0, 8] = float(acc_err_3[2])
        if (self.if_rnn_network == False):
            self.torch_input_vec = torch.from_numpy(self.nn_input_vec).float()
            self.torch_output_vec = self.control_network(self.torch_input_vec)
            self.nn_output_vec = self.torch_output_vec.data.numpy().tolist()[0]
        else:
            self.torch_input_vec = torch.from_numpy(self.nn_input_vec[np.newaxis, :, :]).float()
            self.torch_output_vec, hidden_state = self.control_network(self.torch_input_vec, self.hidden_state)
            self.hidden_state = hidden_state.data
            self.nn_output_vec = self.torch_output_vec.data.numpy()[0, :, :].tolist()[0]

        # print(self.torch_input_vec.shape,self.torch_input_vec)

        yaw_rate = self.compute_yaw_rate()
        r, p, y, throttle = self.nn_output_vec

        # print(r * 57.3, p * 57.3, y * 57.3, throttle, yaw_rate)
        self.client.moveByAngleThrottleAsync(p, r, throttle, yaw_rate, 3e8)
        # self.client.moveByAngleThrottleAsync(, float(self.nn_output_vec[0]), float(self.nn_output_vec[3]), yaw_rate ,3e8)

        if (self.if_log):
            self.logger.update("time_stamp", self.current_time)
            self.logger.update("current_pos", self.current_pos)
            self.logger.update("current_spd", self.current_spd)
            self.logger.update("current_acc", self.current_acc)
            self.logger.update("current_rot", self.current_rot)

            self.logger.update("target_pos", self.target_pos)
            self.logger.update("target_spd", self.target_spd)
            self.logger.update("target_acc", self.target_acc)
            self.logger.update("target_rot", self.target_rot)

            self.logger.update("input_roll", r)
            self.logger.update("input_pitch", p)
            self.logger.update("input_yaw", y)
            self.logger.update("input_throttle", throttle)
        pass

    def follow_rapid_trajectory(self, input_vec_np, start_pos):
        # print("------------------")
        self.refresh_state()
        self.current_time = (cv2.getTickCount() - self.t_start) / cv2.getTickFrequency()
        torch_input = torch.from_numpy(input_vec_np.T).float()
        # print(torch_input.shape, torch_input)

        plan_network_output = self.plan_network(torch_input).data.numpy()

        self.target_pos = plan_network_output[:, (0, 1, 2)].T + start_pos
        self.target_spd = plan_network_output[:, (3, 4, 5)].T
        self.target_acc = plan_network_output[:, (6, 7, 8)].T - np.matrix([0, 0, 9.8]).T

        pos_err_3 = self.target_pos - self.current_pos
        spd_err_3 = self.target_spd - self.current_spd
        acc_err_3 = self.target_acc - self.current_acc

        self.nn_input_vec[0, 0] = float(pos_err_3[0])
        self.nn_input_vec[0, 1] = float(pos_err_3[1])
        self.nn_input_vec[0, 2] = float(pos_err_3[2])
        self.nn_input_vec[0, 3] = float(spd_err_3[0])
        self.nn_input_vec[0, 4] = float(spd_err_3[1])
        self.nn_input_vec[0, 5] = float(spd_err_3[2])
        self.nn_input_vec[0, 6] = float(acc_err_3[0])
        self.nn_input_vec[0, 7] = float(acc_err_3[1])
        self.nn_input_vec[0, 8] = float(acc_err_3[2])

        # print(self.nn_input_vec.T)
        if (self.if_rnn_network == False):
            self.torch_input_vec = torch.from_numpy(self.nn_input_vec).float()
            self.torch_output_vec = self.control_network(self.torch_input_vec)
            self.nn_output_vec = self.torch_output_vec.data.numpy().tolist()[0]
        else:
            self.torch_input_vec = torch.from_numpy(self.nn_input_vec[np.newaxis, :, :]).float()
            self.torch_output_vec, hidden_state = self.control_network(self.torch_input_vec, self.hidden_state)
            self.hidden_state = hidden_state.data
            self.nn_output_vec = self.torch_output_vec.data.numpy()[0, :, :].tolist()[0]
        pass

        yaw_rate = self.compute_yaw_rate()
        r, p, y, throttle = self.nn_output_vec

        # print(r * 57.3, p * 57.3, y * 57.3, throttle, yaw_rate)
        self.client.moveByAngleThrottleAsync(p, r, throttle, yaw_rate, 3e8)

        # yaw_rate = self.compute_yaw_rate()
        # r, p, y, throttle = self.nn_output_vec

        # print(r * 57.3, p * 57.3, y * 57.3, throttle, yaw_rate)
        # self.client.moveByAngleThrottleAsync(p, r, throttle, yaw_rate, 3e8)

        if (self.if_log):
            self.logger.update("time_stamp", self.current_time)
            self.logger.update("current_pos", self.current_pos)
            self.logger.update("current_spd", self.current_spd)
            self.logger.update("current_acc", self.current_acc)
            self.logger.update("current_rot", self.current_rot)

            self.logger.update("target_pos", self.target_pos)
            self.logger.update("target_spd", self.target_spd)
            self.logger.update("target_acc", self.target_acc)
            self.logger.update("target_rot", self.target_rot)

            self.logger.update("input_roll", r)
            self.logger.update("input_pitch", p)
            self.logger.update("input_yaw", y)
            self.logger.update("input_throttle", throttle)
        pass
