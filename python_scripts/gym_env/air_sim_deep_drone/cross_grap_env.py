"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import airsim
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import transforms3d
import time, os, sys

WORK_DIR = os.getenv("CROSS_GAP_WORK_DIR")
if WORK_DIR is None:
    WORK_DIR = "../"
sys.path.append("%s/query_data"%WORK_DIR)
sys.path.append("%s/query_data/tools"%WORK_DIR)
sys.path.append("%s/deep_drone"%WORK_DIR)
sys.path.append("%s"%WORK_DIR)
from narrow_gap import narrow_gap
import cv2
# action: roll, pitch, yaw, throttle
# observation: pos_3, spd_3, acc_3, rot_3


def colorize(string, color, bold=False, highlight=False):
    color2num = dict(
            gray=30,
            red=31,
            green=32,
            yellow=33,
            blue=34,
            magenta=35,
            cyan=36,
            white=37,
            crimson=38
    )
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Cross_gap_env(gym.Env):

    def plot_log(self):
        import matplotlib.pyplot as plt
        t = self.observation_log[range(0, self.log_idx), 12]
        pos_x = self.observation_log[range(0, self.log_idx), 0]
        pos_y = self.observation_log[range(0, self.log_idx), 1]
        pos_z = self.observation_log[range(0, self.log_idx), 2]

        spd_x = self.observation_log[range(0, self.log_idx), 3]
        spd_y = self.observation_log[range(0, self.log_idx), 4]
        spd_z = self.observation_log[range(0, self.log_idx), 5]

        acc_x = self.observation_log[range(0, self.log_idx), 6]
        acc_y = self.observation_log[range(0, self.log_idx), 7]
        acc_z = self.observation_log[range(0, self.log_idx), 8]

        action_r = self.action_log[range(0, self.log_idx), 0]
        action_p = self.action_log[range(0, self.log_idx), 1]
        action_y = self.action_log[range(0, self.log_idx), 2]
        action_thr = self.action_log[range(0, self.log_idx), 3]

        reward_vec = self.reward_log[range(0,self.log_idx), 0]
        print(colorize("===== Plot log =====", "blue"))
        print(colorize("===== size = %d ==="%self.log_idx, "blue"))

        fig = plt.figure("crossgap_env_observation")
        plt.clf()
        plt.suptitle(self.log_str)
        fig.set_size_inches(16 * 2, 9 * 2)
        plt.subplot(4, 1, 1)
        plt.plot(t,pos_x, 'r', label="pos_x" )
        plt.plot(t,pos_y, 'k', label="pos_y")
        plt.plot(t,pos_z, 'b', label="pos_z")
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(t,spd_x, 'r', label="spd_x" )
        plt.plot(t,spd_y, 'k', label="spd_y")
        plt.plot(t,spd_z, 'b', label="spd_z")
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(t,acc_x, 'r', label="acc_x" )
        plt.plot(t,acc_y, 'k', label="acc_y")
        plt.plot(t,acc_z, 'b', label="acc_z")
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(t, reward_vec, 'r', label="reward" )

        plt.grid()
        plt.legend()

        plt.pause(0.1)

        fig.savefig("%s/test_state_%d_%d.png" % ("./save_img", self.sim_times, int(self.sum_reward)))

        fig = plt.figure("crossgap_env_action")
        plt.clf()
        plt.suptitle(self.log_str)
        fig.set_size_inches(16 * 2, 9 * 2)
        plt.subplot(3, 1, 1)
        plt.plot(action_r, 'r', label="roll")
        plt.plot(action_p, 'k', label="pitch")
        plt.plot(action_y, 'b', label="yaw")
        plt.legend()
        plt.grid()
        plt.pause(0.1)

        plt.subplot(3, 1, 2)
        plt.plot(action_thr, 'k', label="throttle")
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(t, reward_vec, 'r', label="reward" )

        plt.grid()
        plt.legend()
        plt.pause(0.1)
        fig.savefig("%s/test_ctrl_%d_%d.png" % ("./save_img", self.sim_times, int(self.sum_reward)))

    def test(self):
        print("Hello， this is test function, V0.0.2")

    def init_airsim_client(self):
        self.client = airsim.MultirotorClient(ip="127.0.0.1", timeout_value=5)
        self.initialized = 0
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.takeoffAsync(1)
            self.client.hoverAsync()
            self.initialized = 1
            self.air_sim_vehicle_pose = self.client.simGetVehiclePose()
        except Exception as e:
            print(e)
            print(colorize("===== Connect error ====", 'red'))
            time.sleep(5)
            self.initialized = 0
        if self.initialized == 1:
            self.get_state()
        else:
            self.init_airsim_client()

    def compute_yaw_rate(self):
        cur_roll_vec = self.current_rot[:, 0]
        tar_roll_vec = self.target_rot[:, 0]
        current_yaw = math.atan2(cur_roll_vec[1], cur_roll_vec[0])
        target_yaw = math.atan2(tar_roll_vec[1], tar_roll_vec[0])

        return -self.control_amp * 1 * (2 * (target_yaw - current_yaw) - self.control_amp * 0.1 * (-1 * self.raw_state.kinematics_estimated.angular_acceleration.z_val))

    def print_help(self):
        # version = "V2.0"
        # desc = "A stable version, can recover anyway"
        version = "V1.0"
        desc = "Add crossgap_env"
        print(colorize("===========================================================", 'red'))
        print(colorize("This is %s" % self.name, 'white'))
        print(colorize("===== Version %s=====" % version, 'yellow'))
        print(colorize("===== Desc %s=====" % desc, 'yellow'))
        print(colorize("Observation_29 = [current_pos_3, current_spd_3, current_acc_3, current_rot_3, t_1, cross_spd_1, pos_err_3, spd_start_3, acc_start_3, spd_end_3, acc_end_3]", 'blue'))
        print(colorize("Action_4 = [roll, pitch, yaw, throttle]", 'magenta'))
        print(colorize("===========================================================", 'red'))

    def __init__(self):
        np.set_printoptions(precision=2)
        self.game_mode = "no_bias"
        self.name = "Crossgap-v0"
        self.sim_times = 0
        self.rad2deg = 180.0 / math.pi
        self.coommand_last_time = 0.5

        self.if_debug = 1
        try:
            os.mkdir('./save_img/')
        except Exception as e:
            pass

        self.print_help()

        self.transform_R_world_to_ned = np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).T

        self.control_amp = 1
        self.Kp = self.control_amp * 6 * np.eye(3, 3)
        self.Kv = self.control_amp * 6 * np.eye(3, 3)

        self.start_pos = [-10, 10, 10]
        self.pos_bias = np.matrix([ [self.start_pos[0]],[self.start_pos[1]],[self.start_pos[2]] ] )
        # print(self.env_start_pos.shape, self.env_start_pos)

        plane_size = -0.00
        rad2angle = 180.0 / math.pi
        self.narrow_gap = narrow_gap([10 - plane_size, 0, 5 - plane_size], [0 / rad2angle, 0 / rad2angle, -30 / rad2angle]);
        self.narrow_gap.para_cross_spd = 4.0
        self.narrow_gap.cross_ballistic_trajectory(if_draw=0);
        self.narrow_gap.approach_trajectory(self.start_pos, [0, 0, 0], [0, 0, 0], if_draw=0)
        self.need_replan = 1
        # action: roll, pitch, yaw, throttle
        max_angle = 30.0
        action_low_bound = np.array([-max_angle / self.rad2deg, -max_angle / self.rad2deg, -max_angle / self.rad2deg, 0])
        action_up_bound = np.array([max_angle / self.rad2deg, max_angle / self.rad2deg, max_angle / self.rad2deg, 1])

        # observation: pos_3, spd_3, acc_3, rot_3


        max_time = 6
        max_cross_spd = 10
        max_obs_pos = 20.0
        max_obs_spd = 20.0
        max_obs_acc = 20.0
        max_obs_euler = 180.0
        obs_up_bound = np.array([max_obs_pos, max_obs_pos, max_obs_pos,
                                 max_obs_spd, max_obs_spd, max_obs_spd,
                                 max_obs_acc, max_obs_acc, max_obs_acc,
                                 max_obs_euler, max_obs_euler, max_obs_euler,
                                 max_time, max_cross_spd,
                                 max_obs_pos, max_obs_pos, max_obs_pos,
                                 max_obs_spd, max_obs_spd, max_obs_spd,
                                 max_obs_acc, max_obs_acc, max_obs_acc,
                                 max_obs_spd, max_obs_spd, max_obs_spd,
                                 max_obs_acc, max_obs_acc, max_obs_acc])
        obs_low_bound = -obs_up_bound

        # print(action_low_bound)
        # print(action_up_bound)
        self.sum_reward = 0.0
        self.action_space = spaces.Box(low=action_low_bound, high=action_up_bound, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low_bound, high=obs_up_bound, dtype=np.float32)
        self.observation = np.zeros(29, dtype=np.float32)

        # Log module
        self.if_log = 0
        self.log_str = ""
        self.observation_log = np.zeros([10000, 29], dtype=np.float32)
        self.action_log = np.zeros([10000, 4], dtype=np.float32)
        self.reward_log = np.zeros([10000, 1], dtype=np.float32)
        self.last_acc = np.matrix([0, 0, 0]).T
        self.log_idx = 0

        # history information
        self.last_response_time  = cv2.getTickCount()
        self.last_ctrl = np.matrix([0,0,0,0])
        self.current_ctrl=  np.matrix([0,0,0,0])

        # self.observation_space = spaces.Box(low=-10000, high=10000, shape=(12,), dtype=np.float32)
        try:
            self.log_txt = open('c:/deep_drone_log.txt','w')
        except :
            self.log_txt = open('./deep_drone_log.txt','w')
        self.first_run = 1
        self.trajectory_start_t = 0
        self.init_airsim_client()
        self.reset()

        self.if_log = 1

    def ned_to_world(self, vector):
        return self.transform_R_world_to_ned.T * vector

    def replan(self):
        print(colorize("===== Replan  =====","red"))

        if self.game_mode == "with_bias":
            start_pos = self.current_pos
        else:
            start_pos = self.current_pos + self.pos_bias
        start_pos = self.start_pos
        self.narrow_gap.approach_trajectory(start_pos,
                                       self.current_spd,
                                       self.current_acc, if_draw=0)

        self.narrow_gap.cross_ballistic_trajectory(if_draw=0);

        self.approach_time = self.narrow_gap.Tf
        self.cross_time = self.narrow_gap.t_c

        final_target_rot = self.narrow_gap.get_rotation(self.approach_time + self.cross_time)
        final_target_rot = self.transform_R_world_to_ned*final_target_rot
        if (np.linalg.det(final_target_rot) == -1):
            final_target_rot[:, 1] = final_target_rot[:, 1] * -1
        self.final_rpy = transforms3d.euler.mat2euler( final_target_rot, axes = 'sxyz')

        self.trajectory_start_t = cv2.getTickCount()
    # Get observation
    def get_state(self):


        state = self.client.getMultirotorState()
        self.raw_state = state
        if self.game_mode == "with_bias":
            print("crossgap output with bias")
            self.current_pos = self.ned_to_world(np.matrix([state.kinematics_estimated.position.x_val,
                                                            state.kinematics_estimated.position.y_val,
                                                            state.kinematics_estimated.position.z_val]).T)  + self.pos_bias
        else:
            # print("self.narrow_gap.rapid_pos_start = ", self.narrow_gap.rapid_pos_start)
            self.current_pos = self.ned_to_world(np.matrix([state.kinematics_estimated.position.x_val,
                                                            state.kinematics_estimated.position.y_val,
                                                            state.kinematics_estimated.position.z_val]).T)
                                                            # state.kinematics_estimated.position.z_val]).T) + self.pos_bias - self.narrow_gap.rapid_pos_start

        self.current_spd = self.ned_to_world(np.matrix([state.kinematics_estimated.linear_velocity.x_val,
                                                        state.kinematics_estimated.linear_velocity.y_val,
                                                        state.kinematics_estimated.linear_velocity.z_val]).T)

        self.current_acc = self.ned_to_world(np.matrix([state.kinematics_estimated.linear_acceleration.x_val,
                                                        state.kinematics_estimated.linear_acceleration.y_val,
                                                        state.kinematics_estimated.linear_acceleration.z_val]).T)

        self.current_angular_vel = self.ned_to_world(np.matrix([state.kinematics_estimated.angular_velocity.x_val,
                                                        state.kinematics_estimated.angular_velocity.y_val,
                                                        state.kinematics_estimated.angular_velocity.z_val]).T)

        self.current_angular_acc = self.ned_to_world(np.matrix([state.kinematics_estimated.angular_acceleration.x_val,
                                                        state.kinematics_estimated.angular_acceleration.y_val,
                                                        state.kinematics_estimated.angular_acceleration.z_val]).T)

        ori_q = state.kinematics_estimated.orientation
        q = [ori_q.w_val, ori_q.x_val, ori_q.y_val, ori_q.z_val]

        self.current_rot = self.transform_R_world_to_ned.T * transforms3d.euler.quat2mat(q)
        self.current_euler = transforms3d.euler.quat2euler(q, 'sxyz')


        self.observation[range(0, 3)] = np.array(self.current_pos).flatten()
        self.observation[range(3, 6)] = np.array(self.current_spd).flatten()
        self.observation[range(6, 9)] = np.array(self.current_acc).flatten()
        self.observation[range(9, 12)] = np.array(self.current_euler).flatten() * self.rad2deg

        if (self.need_replan or
            (((cv2.getTickCount() - self.last_response_time) / cv2.getTickFrequency()) > 1)
            ):
            self.last_response_time = cv2.getTickCount()
            self.need_replan = 0
            self.get_state()
            self.replan()
            for i in range(17 - 1):
                self.observation[i + 13] = self.narrow_gap.rapid_nn_input_vector[i + 1]

        self.current_time = (cv2.getTickCount() - self.trajectory_start_t) / cv2.getTickFrequency()
        self.observation[12] = self.current_time
        self.state = self.observation
        self.last_response_time = cv2.getTickCount()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_function(self):
        if self.game_mode == "with_bias":
            current_pos = self.current_pos
        else:
            # print("self.current_pos = ", self.current_pos.T)
            current_pos = self.current_pos + self.pos_bias
            # print("current_pos = ", current_pos.T)

        distance = current_pos.T - np.matrix(self.narrow_gap.center).reshape([1, 3])
        # print("current_pos = ", current_pos.T, " center = ", np.matrix(self.narrow_gap.center), '   dis = ', distance, ' mod = ', np.linalg.norm(distance))
        # return max(10.0 - np.linalg.norm(distance),0)
        # print("Distance = %.1f "% np.linalg.norm(distance))
        reward_obs = (1.0- np.linalg.norm(distance)) * 10000  # obs reward
        if reward_obs < 0:
            reward_obs = 0
        else:
            print("Bingo")

        # spd positive reward
        # reward_obs = reward_obs + np.linalg.norm(self.current_spd)*10

        # reward_ctrl = (-1.0 * np.linalg.norm(self.current_spd)) + (-1.0 * np.linalg.norm(self.current_acc))  + (-0.01*np.linalg.norm(self.current_angular_vel)*self.rad2deg) # obs reward

        # angular velocity penalty
        reward_obs = reward_obs + (0.2 * (2.0 - np.linalg.norm(self.current_angular_vel) * self.rad2deg)) + \
                      (0.5 * (2.0 - np.linalg.norm(self.current_angular_acc) * self.rad2deg))     # obs reward



        # print("acc_diff = ", 10*np.linalg.norm(self.current_acc - self.last_acc))
        reward_obs =  reward_obs - 1.0*np.linalg.norm(self.current_acc - self.last_acc)

        self.last_acc = self.current_acc
        # reward_ctrl
            # control different penalty
        # print("current_ctrl = ", self.current_ctrl, "last_ctrl = ", self.last_ctrl)
        # reward_ctrl = reward_ctrl*1.0 + (0.1 - np.linalg.norm(self.current_ctrl[0,3] - self.last_ctrl[0,3])) * 10000.0
        reward_ctrl = 0
        reward_ctrl = 0.5*reward_ctrl
        reward_total = reward_obs + reward_ctrl
        ## print("rw_total = %.1f, rw_obs = %.1f, rw_ctrl = %.1f"%( reward_total,reward_obs,  reward_ctrl))
        return reward_total

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.step_count = self.step_count + 1
        self.last_ctrl = self.current_ctrl
        roll, pitch, yaw, throttle = action.flatten()
        # print("action = ", roll, pitch, yaw, throttle )
        self.target_rot = transforms3d.euler.euler2mat(roll, pitch, yaw, axes='szyx')
        yaw_rate = self.compute_yaw_rate()

        done = 0
        try:
            self.get_state()

            if (self.current_time > self.approach_time ):
                roll = self.final_rpy[0]
                pitch = self.final_rpy[1]
                yaw = self.final_rpy[2]
                throttle = 0.4
            self.current_ctrl = np.matrix([roll, pitch, yaw, throttle])
            reward = self.reward_function()

            self.client.moveByAngleThrottleAsync(float(pitch), float(roll), float(throttle), float(yaw_rate), self.coommand_last_time)
            collision = self.client.simGetCollisionInfo()

            if ((self.current_time > 2 * self.cross_time + self.approach_time) or
                    self.step_count > 10000 or
                    len(collision.object_name) > 1):
                done = 1
                # self.client.hoverAsync()

        except Exception as e:
            print(colorize("===== Error in simulate step =====", "red"))
            print("error = ",e)
            done = 1
            reward = 0
            return np.array(self.state), reward, done, {}

        if(done == 1):
            if(len(collision.object_name) > 1 ):
                reward = -1000000
            else:
                reward = 0
            self.log_str = "sim_time = %d, step = %d, r_sum = %.2f, collisied = %d" % (self.sim_times, self.step_count, self.sum_reward+reward, len(collision.object_name) > 1)
            if (self.if_debug):
                print(colorize(self.log_str, 'red'))
            self.log_txt.write(self.log_str + '\n')
            self.log_txt.flush()

        self.sum_reward = self.sum_reward + reward
        if(self.if_log):
            obs = np.array(self.state).reshape(29).tolist()
            self.observation_log[self.log_idx, :] = obs
            self.action_log[self.log_idx, :] = [roll, pitch, yaw, throttle]
            self.reward_log[self.log_idx, :] = [reward]
            self.log_idx = self.log_idx + 1

        # print('self.state = ', np.array(self.state))
        # print("gym = ", np.array(self.state))
        return np.array(self.state), reward, done, {}

    def reset(self):
        del self.client
        self.init_airsim_client()
        try:
            # def simSetVehiclePose(self, pose, ignore_collison, vehicle_name=''):
            # def simGetVehiclePose(self, vehicle_name=''):
            # def simGetObjectPose(self, object_name):
            self.client.enableApiControl(True)
            self.client.reset()

            # position = airsim.Vector3r(self.env_start_pos[0], self.env_start_pos[1], self.env_start_pos[2])
            # heading = airsim.utils.to_quaternion(0, 0, 0)
            # pose = airsim.Pose(position, heading)
            # self.client.simSetVehiclePose(pose, ignore_collison=True)
            if(self.first_run == 1):
                # self.client.simSetVehiclePose(pose, ignore_collison=False)
                self.first_run = 0
            self.client.enableApiControl(True)
            self.client.armDisarm(True)

            # self.client.simSetVehiclePose(pose=self.air_sim_vehicle_pose, ignore_collison=1)
            self.client.enableApiControl(True)
            self.client.takeoffAsync(1)
            self.client.hoverAsync()
            self.get_state()
            self.need_replan = 1

            # self.client.simPause()
        except Exception as e:
            print(e)
            print(colorize("===== Error in reset =====","red"))
            self.init_airsim_client()
            self.reset()

        if (self.if_log):
            self.plot_log()
        self.log_idx = 0
        self.step_count = 0
        self.sum_reward = 0.0
        self.sim_times = self.sim_times + 1
        self.client.moveByAngleThrottleAsync(0,0,0.6,0,3e8)
        time.sleep(1)
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        pass

    def close(self):
        self.client.enableApiControl(False)
        del self.client
