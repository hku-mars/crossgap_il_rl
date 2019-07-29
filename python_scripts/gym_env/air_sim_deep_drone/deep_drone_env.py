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
import time

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


class Deep_drone_env(gym.Env):

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
        version = "V2.1"
        desc = "Add angular vel"
        print(colorize("===========================================================", 'red'))
        print(colorize("This is %s" % self.name, 'white'))
        print(colorize("===== Version %s=====" % version, 'yellow'))
        print(colorize("===== Desc %s=====" % desc, 'yellow'))
        print(colorize("Observation_12 = [pos_3, spd_3, acc_3, rot_3]", 'blue'))
        print(colorize("Action_4 = [roll, pitch, yaw, throttle]", 'magenta'))
        print(colorize("===========================================================", 'red'))

    def __init__(self):
        self.name = "Deepdrone-v0"
        self.sim_times = 0
        self.rad2deg = 180.0 / math.pi

        self.if_debug = 1
        self.print_help()

        self.transform_R_world_to_ned = np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).T
        self.init_airsim_client()

        self.reset()
        self.control_amp = 1
        self.Kp = self.control_amp * 6 * np.eye(3, 3)
        self.Kv = self.control_amp * 6 * np.eye(3, 3)

        # action: roll, pitch, yaw, throttle

        max_angle = 30.0
        action_low_bound = np.array([-max_angle / self.rad2deg, -max_angle / self.rad2deg, -max_angle / self.rad2deg, 0])
        action_up_bound = np.array([max_angle / self.rad2deg, max_angle / self.rad2deg, max_angle / self.rad2deg, 1])

        # observation: pos_3, spd_3, acc_3, rot_3

        max_obs_pos = 10.0
        max_obs_spd = 20.0
        max_obs_acc = 20.0
        max_obs_euler = 180.0
        obs_up_bound = np.array([max_obs_pos, max_obs_pos, max_obs_pos,
                                 max_obs_spd, max_obs_spd, max_obs_spd,
                                 max_obs_acc, max_obs_acc, max_obs_acc,
                                 max_obs_euler, max_obs_euler, max_obs_euler])
        obs_low_bound = -obs_up_bound

        # print(action_low_bound)
        # print(action_up_bound)
        self.sum_reward = 0.0
        self.action_space = spaces.Box(low=action_low_bound, high=action_up_bound, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low_bound, high=obs_up_bound, dtype=np.float32)
        # self.observation_space = spaces.Box(low=-10000, high=10000, shape=(12,), dtype=np.float32)
        self.log_txt = open('c:/deep_drone_log.txt','w')

    def ned_to_world(self, vector):
        return self.transform_R_world_to_ned.T * vector

    # Get observation
    def get_state(self):
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

        self.current_angular_vel = self.ned_to_world(np.matrix([state.kinematics_estimated.angular_velocity.x_val,
                                                        state.kinematics_estimated.angular_velocity.y_val,
                                                        state.kinematics_estimated.angular_velocity.z_val]).T)

        ori_q = state.kinematics_estimated.orientation
        q = [ori_q.w_val, ori_q.x_val, ori_q.y_val, ori_q.z_val]

        self.current_rot = self.transform_R_world_to_ned.T * transforms3d.euler.quat2mat(q)
        self.current_euler = transforms3d.euler.quat2euler(q, 'sxyz')

        self.observation = np.zeros(12, dtype=np.float32)

        self.observation[range(0, 3)] = np.array(self.current_pos).flatten()
        self.observation[range(3, 6)] = np.array(self.current_spd).flatten()
        self.observation[range(6, 9)] = np.array(self.current_acc).flatten()
        self.observation[range(9, 12)] = np.array(self.current_euler).flatten()*self.rad2deg

        self.state = self.observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_function(self):
        distance = self.current_pos.T - np.array([0, 0, 5])
        # print("current_pos = ", self.current_pos.T, '   dis = ', distance, ' mod = ',np.linalg.norm(distance) )
        # return max(10.0 - np.linalg.norm(distance),0)
        reward_obs = 10.0 - np.linalg.norm(distance)  # obs reward

        reward_ctrl = (-1.0 * np.linalg.norm(self.current_spd)) + (-1.0 * np.linalg.norm(self.current_acc))  + (-0.01*np.linalg.norm(self.current_angular_vel)*self.rad2deg) # obs reward
        reward_ctrl = 0.5*reward_ctrl
        reward_total = reward_obs + reward_ctrl
        ## print("rw_total = %.1f, rw_obs = %.1f, rw_ctrl = %.1f"%( reward_total,reward_obs,  reward_ctrl))
        return reward_total

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.step_count = self.step_count + 1

        roll, pitch, yaw, throttle = action.flatten()
        # print("action = ", roll, pitch, yaw, throttle )
        self.target_rot = transforms3d.euler.euler2mat(roll, pitch, yaw, axes='szyx')
        yaw_rate = self.compute_yaw_rate()

        reward = self.reward_function()
        done = 0

        try:
            self.client.moveByAngleThrottleAsync(float(pitch), float(roll), float(throttle), float(yaw_rate), 3e8)
            self.get_state()
            collision = self.client.simGetCollisionInfo()
        except Exception as e:
            print(colorize("===== Error in simulate step =====", "red"))
            done = 1
            reward = 0
            return np.array(self.state), reward, done, {}

        if (len(collision.object_name) > 1 or
                self.step_count > 1000):
            log_str = "sim_time = %d, step = %d, r_sum = %.2f, collisied = %d" % (self.sim_times, self.step_count, self.sum_reward, len(collision.object_name) > 1)
            if(self.if_debug):
                print(colorize(log_str, 'red'))
            self.log_txt.write(log_str+'\n')
            self.log_txt.flush()
            done = 1
            reward = 0

        self.sum_reward = self.sum_reward + reward
        # print('self.state = ', np.array(self.state))
        return np.array(self.state), reward, done, {}

    def reset(self):
        del self.client
        self.init_airsim_client()
        try:
            self.client.enableApiControl(True)
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.takeoffAsync(1)
            self.client.hoverAsync()
            self.get_state()
        except Exception as e:
            print(e)
            print(colorize("===== Error in reset =====","red"))
            self.init_airsim_client()
            self.reset()

        self.step_count = 0
        self.sum_reward = 0.0
        self.sim_times = self.sim_times + 1
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        pass

    def close(self):
        self.client.enableApiControl(False)
        del self.client
