import numpy as np
import cv2
import transforms3d
#--- Gym deep_drone env ---
import gym
import air_sim_deep_drone
#--- Gym deep_drone env ---
import math
from tools.plane_logger import plane_logger

import sys, os, time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU using
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append("..")  # Adds higher directory to python modules path.
from pid_controller import Pid_controller

def refresh_quad_rotor_state_kernel(arg, **kwarg ):
    while(1):
        print("run.")
        time.sleep(0.1)
        # Quad_rotor_controller.refresh_state(arg)

# class Quad_rotor_controller(Process):
class Quad_rotor_controller():

    def __init__(self, logger_name="plane_log.txt", if_network=0):
        np.set_printoptions(precision=2)
        # self.env = gym.make('Deepdrone-v')
        self.env = gym.make('Crossgap-v2')
        self.env.if_debug = 0
        self.env.if_log = 0
        # self.env.print_help()
        self.env.reset()
        # super(Quad_rotor_controller, self).__init__()
        self.if_debug = 0
        self.if_log = 1
        self.if_save_log = 1
        self.if_rnn_network = 1
        self.rad2deg = 180.0 / math.pi
        self.logger = plane_logger(logger_name)

        self.pid_ctrl = Pid_controller()

        self.current_pos = self.env.current_pos
        self.current_spd = self.env.current_spd
        self.current_acc = self.env.current_acc
        self.current_rot = np.eye(3, 3)

        self.target_pos = np.zeros([3, 1])
        self.target_spd = np.zeros([3, 1])
        self.target_acc = np.zeros([3, 1])
        self.target_rot = np.eye(3, 3)
        # print(self.target_rot)

        self.target_force = np.zeros([3, 1])
        self.target_force_max = 5
        self.target_roll_vec_preference = np.matrix([1, 0, 0]).T
        self.target_roll_vec = np.matrix([1, 0, 0]).T
        self.target_pitch_vec = np.matrix([0, 1, 0]).T
        self.target_yaw_vec = np.matrix([0, 0, 1]).T

        self.transform_R_world_to_ned = np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).T
        self.control_amp = 1
        # self.Kp = self.control_amp * 20 * np.eye(3, 3)
        # self.Kv = self.control_amp * 20 * np.eye(3, 3)

        self.Kp = self.control_amp * 6 * np.eye(3, 3)
        self.Kv = self.control_amp * 6 * np.eye(3, 3)

        self.gravity = np.array([[0], [0], [-9.8]])
        self.quad_painter = []
        self.t_start = cv2.getTickCount()
        self.if_network = if_network
        # self.start_service()
        if(self.if_network):
            self.build_tf_net()

        self.gym_obs = np.zeros([1, 12])
        print("Init finish")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.close()
        print("release quad_rotor_controller")

    def hover(self):
        self.env.client.hoverAsync()

    def build_tf_net(self):
        import tensorflow as tf
        from tf_policy_network import Policy_network
        self.obervation_29 = tf.placeholder(shape=[None, 29], dtype= tf.float32)
        self.policy_network = Policy_network(self.obervation_29)
        self.sess = self.policy_network.sess
        self.net_plane_state = self.policy_network.net_plane_state
        self.control_network = self.policy_network.control_network
        self.plan_network = self.policy_network.plan_network
        if(self.policy_network.if_mode_rnn):
            self.feed_hidden_state_0 = self.policy_network.feed_hidden_state_0
            self.feed_hidden_state_1 = self.policy_network.feed_hidden_state_1

        self.nn_input_vec =  self.policy_network.nn_input_vec
        self.nn_output_vec = self.policy_network.nn_output_vec
        self.nn_output_vec_last = self.policy_network.nn_output_vec

    def save_log(self):
        if (self.if_save_log):
            self.logger.save()

    def init_airsim_client(self):
        self.env.reset()

    def reset_client(self):
        self.env.reset()

    def set_target_body(self, target_pos, target_spd, target_acc, target_rot):
        self.target_pos = np.matrix(target_pos).T
        self.target_spd = np.matrix(target_spd).T
        self.target_acc = np.matrix(target_acc).T

    def move_to_pos(self, pos, if_enable_network=1):
        # self.if_log = 0
        print("Move to pos:", pos)
        self.set_target_body(pos, [0, 0, 0], [0, 0, 0], np.eye(3, 3))
        max_pos_err = 0.5
        max_spd = 0.5
        max_acc = 0.5
        while (1):
            self.env.game_mode = "with_bias"
            self.refresh_state()
            if (self.if_network and if_enable_network):
                # self.control()
                self.control_using_network()
            else:
                print("classic control does not supprot !!!")
                # self.control()
            pos_err = self.target_pos - self.current_pos
            spd_err = self.target_spd - self.current_spd
            acc_err = self.target_acc - self.current_acc
            if (np.linalg.norm(pos_err) < max_pos_err):
                # print(pos_err)
                if (np.linalg.norm(spd_err) < max_spd):
                    if (np.linalg.norm(acc_err) < max_acc):
                        break
        print("Move to pos:", pos, "finish")

    def contorl_attitude(self, rot):
        # self.env.target_rot =  rot
        self.env.target_rot  = self.transform_R_world_to_ned *rot
        # This is very important, if the determing be minues one, something error can be happened.
        if (np.linalg.det(self.env.target_rot) == -1):
            self.env.target_rot[:, 1] = self.env.target_rot[:, 1] * -1
        r, p, y = transforms3d.euler.mat2euler( self.env.target_rot, axes = 'sxyz')
        # print('rpy = ', r * 57.3, p * 57.3, y * 57.3)
        self.gym_obs = self.env.step(np.array([r, p, y, 0.4]))

    def refresh_state(self):
        # print(self.gym_obs)
        self.current_pos = np.matrix([self.gym_obs[0][0],self.gym_obs[0][1],self.gym_obs[0][2]]).T
        self.current_spd = np.matrix([self.gym_obs[0][3],self.gym_obs[0][4],self.gym_obs[0][5]]).T
        self.current_acc = np.matrix([self.gym_obs[0][6],self.gym_obs[0][7],self.gym_obs[0][8]]).T
        self.current_euler = np.matrix([self.gym_obs[0][9], self.gym_obs[0][10], self.gym_obs[0][11]]).T
        self.current_rot = transforms3d.euler.euler2mat(self.gym_obs[0][9],self.gym_obs[0][10],self.gym_obs[0][11], axes='sxyz')

        q = transforms3d.quaternions.mat2quat(self.current_rot)
        self.pid_ctrl.update_state(self.current_pos.ravel().tolist()[0],
                                   self.current_spd.ravel().tolist()[0],
                                   [0, 0, 0], q)

        if(len(self.gym_obs[0])>12): # env with planning
            self.plane_vec = self.gym_obs[0][range(12,29)]
            # print(self.plane_vec)

        if (self.if_debug):
            print("---------------")
            print("Current_pos = ", self.current_pos.T)
            # print("Current_pos = ", self.target_pos.T)
            print("Current_spd = ", self.current_spd.T)
            # print("Current_pos = ", self.target_spd.T)

    def pid_rpy_to_air_sim_rpy(self, res_r, res_p, res_y = 0):

        mat_new_mat = transforms3d.euler.euler2quat(res_y, res_p, res_r, axes='sxyz')
        res_r, res_p, res_y = transforms3d.euler.quat2euler(mat_new_mat, axes='szyx')

        return res_r, -res_p, res_y

    def control_using_network(self):
        self.current_time = (cv2.getTickCount() - self.t_start) / cv2.getTickFrequency()
        pos_err_3 = self.target_pos - self.current_pos
        spd_err_3 = self.target_spd - self.current_spd
        acc_err_3 = self.target_acc - self.current_acc*0

        self.nn_input_vec[0, 0] = float(pos_err_3[0])
        self.nn_input_vec[0, 1] = float(pos_err_3[1])
        self.nn_input_vec[0, 2] = float(pos_err_3[2])
        self.nn_input_vec[0, 3] = float(spd_err_3[0])
        self.nn_input_vec[0, 4] = float(spd_err_3[1])
        self.nn_input_vec[0, 5] = float(spd_err_3[2])
        self.nn_input_vec[0, 6] = float(acc_err_3[0])
        self.nn_input_vec[0, 7] = float(acc_err_3[1])
        self.nn_input_vec[0, 8] = float(acc_err_3[2])
        self.nn_input_vec[0, 9] = float(self.current_euler[0])
        self.nn_input_vec[0, 10] = float(self.current_euler[1])
        self.nn_input_vec[0, 11] = float(self.current_euler[2])
        if (self.policy_network.if_mode_rnn == 0):
            tf_control_network_out = self.sess.run(self.control_network.net_output,feed_dict={self.control_network.net_input: np.matrix(self.nn_input_vec)} )
        else:
            [tf_control_network_out, self.feed_hidden_state_0, self.feed_hidden_state_1] = self.sess.run([self.control_network.net_output,
                                                                                                          self.control_network.rnn_hidden_state_out[0],
                                                                                                          self.control_network.rnn_hidden_state_out[1]],
                                                                                                         feed_dict={self.control_network.net_input:              np.matrix(self.nn_input_vec),
                                                                                                                    self.control_network.rnn_hidden_state_in[0]: self.feed_hidden_state_0,
                                                                                                                    self.control_network.rnn_hidden_state_in[1]: self.feed_hidden_state_1})

        self.nn_output_vec = tf_control_network_out[0].tolist()
        # print(self.torch_input_vec.shape,self.torch_input_vec)
        # print(self.nn_output_vec)
        r, p, throttle = self.nn_output_vec

        # print(r * 57.3, p * 57.3, y * 57.3, throttle)
        self.gym_obs = self.env.step(np.array([r, p, throttle]))

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

    def compute_target_rot(self):
        return

    def control(self):
        self.current_time = (cv2.getTickCount() - self.t_start) / cv2.getTickFrequency()
        self.pid_ctrl.update_target(self.target_pos.ravel().tolist()[0], self.target_spd.ravel().tolist()[0], self.target_acc.ravel().tolist()[0], target_yaw=0)
        self.pid_ctrl.update_control()
        self.gym_obs = self.env.step(np.array([self.pid_ctrl.tar_roll, self.pid_ctrl.tar_pitch, self.pid_ctrl.tar_thrust]))
        self.refresh_state()
        r, p, y = self.pid_rpy_to_air_sim_rpy(self.pid_ctrl.tar_roll, self.pid_ctrl.tar_pitch, self.pid_ctrl.tar_yaw)
        throttle = self.pid_ctrl.tar_thrust/9.8
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

    def follow_rapid_trajectory(self, input_vec_np, start_pos):
        self.current_time = (cv2.getTickCount() - self.t_start) / cv2.getTickFrequency()
        # self.env.if_log = 1
        # plan_network_output = self.plan_network(torch_input).data.numpy()
        # plan_network_output = self.sess.run(self.plan_network.net_output, feed_dict={self.plan_network.net_input: np.matrix(input_vec_np.T)})
        # print('---------')
        # print("Input vec = ", np.matrix(input_vec_np).T)
        # print("Plane_vec = ", np.matrix(self.plane_vec))
        # print("Diff__vec = ", np.matrix(input_vec_np).T - np.matrix(self.plane_vec))

        input_vec_np = np.matrix(self.plane_vec).T
        current_plane_state = np.zeros([12,1], dtype= np.float32)

        pos_err_3 = self.target_pos - self.current_pos
        spd_err_3 = self.target_spd - self.current_spd
        acc_err_3 = self.target_acc - self.current_acc

        current_plane_state[range(0, 3)] = self.current_pos - start_pos
        current_plane_state[range(3, 6)] = self.current_spd
        current_plane_state[range(6, 9)] = self.current_acc*0
        current_plane_state[range(9, 12)] = self.current_euler
        np.set_printoptions(precision=2)
        # print(np.matrix(input_vec_np.T))
        if(self.policy_network.if_mode_rnn):
            [tf_control_network_out,plan_network_output, self.feed_hidden_state_0, self.feed_hidden_state_1] = self.sess.run([self.control_network.net_output,
                                                                                                            self.plan_network.net_output,
                                                                                                            self.control_network.rnn_hidden_state_out[0],
                                                                                                            self.control_network.rnn_hidden_state_out[1]],
                                                                                                            feed_dict={self.plan_network.net_input: np.matrix(input_vec_np.T),
                                                                                                                        self.control_network.rnn_hidden_state_in[0]: self.feed_hidden_state_0,
                                                                                                                        self.control_network.rnn_hidden_state_in[1]: self.feed_hidden_state_1,
                                                                                                                        self.net_plane_state: np.matrix(current_plane_state).T})
        else:
            if 0:

                [tf_control_network_out, plan_network_output] = self.sess.run([self.control_network.net_output,
                                                                            self.plan_network.net_output],
                                                                            feed_dict={self.plan_network.net_input: np.matrix(input_vec_np.T),
                                                                                        self.net_plane_state: np.matrix(current_plane_state).T})
            else:
                self.env.game_mode = 'no_bias'
                # print("start_pos = ", start_pos)
                # print("========")
                # for idx in range(9):
                #     self.gym_obs[0][idx] = current_plane_state[idx]
                # print(self.gym_obs[0][range(0, 9)])
                # print(current_plane_state.T)

                [tf_control_network_out, plan_network_output] = self.sess.run([self.control_network.net_output,
                                                                               self.plan_network.net_output],
                                                                              feed_dict={self.policy_network.observation: np.matrix(self.gym_obs[0])})
                # print("Input rpy = ", self.gym_obs[0][range(9,12)]*57.3)

        self.target_pos = plan_network_output[:, (0, 1, 2)].T + start_pos
        self.target_spd = plan_network_output[:, (3, 4, 5)].T
        # self.target_acc = plan_network_output[:, (6, 7, 8)].T - np.matrix([0, 0, 9.8]).T
        self.target_acc = plan_network_output[:, (6, 7, 8)].T

        self.nn_output_vec = tf_control_network_out[0].tolist()
        r, p, throttle = self.nn_output_vec
        y=0
        # r, p, y, throttle = (np.matrix(self.nn_output_vec)*0.5+np.matrix(self.nn_output_vec_last)*0.5).tolist()[0]
        self.nn_output_vec_last = self.nn_output_vec
        self.gym_obs = self.env.step(np.array([r, p, throttle]))
        self.refresh_state()

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

