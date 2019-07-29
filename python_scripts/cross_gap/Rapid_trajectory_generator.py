# This file is a function to generalize rapid trajectory for training
import pickle, os, time
import numpy as np
import copy
# For plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits import mplot3d

import transforms3d
import math
from numpy import linalg as LA  # For LA.norm
import cv2 as cv2
import sys
sys.path.append("%s/query_data/tools"%os.getenv("CROSS_GAP_WORK_DIR"))

import quadrocoptertrajectory as quadtraj


class Rapid_trajectory_generator():
    def __init__(self):
        self.gravity = [0, 0, -9.80]
        self.rapid_traj = None
        self.input_data_all = None
        self.output_data_all = None

    def generate_trajectory(self, pos_err, spd_start, acc_start,
                            spd_end, acc_end, cross_spd):
        self.pos_err = pos_err
        self.spd_start = spd_start
        self.acc_start = acc_start
        self.spd_end = spd_end
        self.acc_end = acc_end
        self.cross_spd = cross_spd

        self.rapid_traj = quadtraj.RapidTrajectory([0, 0, 0], spd_start, acc_start, self.gravity)
        self.rapid_traj.set_goal_position(pos_err)
        self.rapid_traj.set_goal_velocity(spd_end)
        self.rapid_traj.set_goal_acceleration(acc_end)
        self.Tf = float(np.linalg.norm(pos_err) / cross_spd)

        self.rapid_traj.generate(self.Tf)

    def generate_trajectory_cross_time(self, pos_err, spd_start, acc_start,
                                       spd_end, acc_end, cross_time):
        self.pos_err = pos_err
        self.spd_start = spd_start
        self.acc_start = acc_start
        self.spd_end = spd_end
        self.acc_end = acc_end
        # self.cross_spd = cross_spd

        self.rapid_traj = quadtraj.RapidTrajectory([0, 0, 0], spd_start, acc_start, self.gravity)
        self.rapid_traj.set_goal_position(pos_err)
        self.rapid_traj.set_goal_velocity(spd_end)
        self.rapid_traj.set_goal_acceleration(acc_end)
        self.Tf = float(cross_time)
        self.cross_spd = float(np.linalg.norm(pos_err) / self.Tf )
        self.rapid_traj.generate(self.Tf)

    def generate_trajectory_np_array(self, traj):
        self.pos_err = traj[range(2, 5)]
        self.spd_start = traj[range(5, 8)]
        self.acc_start = traj[range(8, 11)]

        self.spd_end = traj[range(11, 14)]
        self.acc_end = traj[range(14, 17)]
        # self.cross_spd = 0.29051678
        self.cross_spd = traj[1]
        # print("============================")
        # print("cross_spd = ", self.cross_spd)
        # print("Pos_err = ", self.pos_err)
        # print("spd_start = ", self.spd_start)
        # print("acc_start = ", self.acc_start)
        #
        # print("spd_end = ", self.spd_end)
        # print("acc_end = ", self.acc_end)

        self.generate_trajectory(self.pos_err, self.spd_start, self.acc_start,
                                 self.spd_end, self.acc_end, self.cross_spd)
        self.export_data(1000)

    def generate_random_trajectory(self, pos_bound, spd_bound, acc_bound, cross_spd_bound):
        try:
            # pass
            # self.pos_err = [np.random.random(pos_bound), np.random.random(pos_bound),np.random.random(pos_bound)]
            self.pos_err = np.random.uniform(size=3) * (pos_bound[1] - pos_bound[0]) + pos_bound[0]
            self.spd_start = np.random.uniform(size=3) * (spd_bound[1] - spd_bound[0]) + spd_bound[0]
            self.acc_start = np.random.uniform(size=3) * (acc_bound[1] - acc_bound[0]) + acc_bound[0]

            self.spd_end = np.random.uniform(size=3) * (spd_bound[1] - spd_bound[0]) + spd_bound[0]
            self.acc_end = np.random.uniform(size=3) * (acc_bound[1] - acc_bound[0]) + acc_bound[0]

            self.cross_spd = np.random.uniform() * (cross_spd_bound[1] - cross_spd_bound[0]) + cross_spd_bound[0]
            # self.cross_spd = np.random.uniform()* (cross_spd_bound[1] - cross_spd_bound[0]) + cross_spd_bound[0]
            # print("============================")
            # print("cross_spd = ", self.cross_spd)
            # print("Pos_err = ", self.pos_err)
            # print("spd_start = ", self.spd_start)
            # print("acc_start = ", self.acc_start)
            #
            # print("spd_end = ", self.spd_end)
            # print("acc_end = ", self.acc_end)

            # self.generate_trajectory(self.pos_err, self.spd_start, self.acc_start,
            #                          self.spd_end, self.acc_end, self.cross_spd)
            self.generate_trajectory(self.pos_err, self.spd_start, self.acc_start,
                                     self.spd_end, self.acc_end, self.cross_spd)
        except Exception as e:
            print("generate_random_trajectory_err: ", e)

    def generate_random_trajectory_cross_time(self, pos_bound, spd_bound, acc_bound, cross_spd_bound):
        try:
            # pass
            # self.pos_err = [np.random.random(pos_bound), np.random.random(pos_bound),np.random.random(pos_bound)]
            self.pos_err = np.random.uniform(size=3) * (pos_bound[1] - pos_bound[0]) + pos_bound[0]
            self.spd_start = np.random.uniform(size=3) * (spd_bound[1] - spd_bound[0]) + spd_bound[0]
            self.acc_start = np.random.uniform(size=3) * (acc_bound[1] - acc_bound[0]) + acc_bound[0]

            self.spd_end = np.random.uniform(size=3) * (spd_bound[1] - spd_bound[0]) + spd_bound[0]
            self.acc_end = np.random.uniform(size=3) * (acc_bound[1] - acc_bound[0]) + acc_bound[0]

            self.Tf = np.random.uniform() * (cross_spd_bound[1] - cross_spd_bound[0]) + cross_spd_bound[0]
            # self.cross_spd = np.random.uniform()* (cross_spd_bound[1] - cross_spd_bound[0]) + cross_spd_bound[0]
            # print("============================")
            # print("cross_spd = ", self.cross_spd)
            # print("Pos_err = ", self.pos_err)
            # print("spd_start = ", self.spd_start)
            # print("acc_start = ", self.acc_start)
            #
            # print("spd_end = ", self.spd_end)
            # print("acc_end = ", self.acc_end)

            # self.generate_trajectory(self.pos_err, self.spd_start, self.acc_start,
            #                          self.spd_end, self.acc_end, self.cross_spd)
            self.generate_trajectory_cross_time(self.pos_err, self.spd_start, self.acc_start,
                                                self.spd_end, self.acc_end, self.Tf)
        except Exception as e:
            print("generate_random_trajectory_err: ", e)

    def generate_given_data(self, pos_bound = [0,0], spd_bound=[0,0], acc_bound=[0,0], cross_spd_bound=[0,0]):
        try:
            # pass
            # self.pos_err = [np.random.random(pos_bound), np.random.random(pos_bound),np.random.random(pos_bound)]
            self.pos_err = np.array([0.5, 0, 0.5]) + np.random.uniform(size=3) * (pos_bound[1] - pos_bound[0]) + pos_bound[0]
            self.spd_start = np.array([0.0, 0, 0.0]) + np.random.uniform(size=3) * (spd_bound[1] - spd_bound[0]) + spd_bound[0]
            self.acc_start = np.array([0.0, 0, 0.0]) + np.random.uniform(size=3) * (acc_bound[1] - acc_bound[0]) + acc_bound[0]

            self.spd_end = np.array([3.5, 0.00, 0.0]) + np.random.uniform(size=3) * (spd_bound[1] - spd_bound[0]) + spd_bound[0]
            self.acc_end = np.array([0.0, 4.24, -2.45]) + np.random.uniform(size=3) * (acc_bound[1] - acc_bound[0]) + acc_bound[0]

            # self.Tf = 2.5 + np.random.uniform() * (cross_spd_bound[1] - cross_spd_bound[0]) + cross_spd_bound[0]
            self.cross_spd = np.random.uniform()* (cross_spd_bound[1] - cross_spd_bound[0]) + cross_spd_bound[0]
            # print("============================")
            # print("cross_spd = ", self.cross_spd)
            # print("Pos_err = ", self.pos_err)
            # print("spd_start = ", self.spd_start)
            # print("acc_start = ", self.acc_start)
            #
            # print("spd_end = ", self.spd_end)
            # print("acc_end = ", self.acc_end)

            # self.generate_trajectory(self.pos_err, self.spd_start, self.acc_start,
            #                          self.spd_end, self.acc_end, self.cross_spd)
            self.generate_trajectory(self.pos_err, self.spd_start, self.acc_start,
                                     self.spd_end, self.acc_end, self.cross_spd)
        except Exception as e:
            print("generate_random_trajectory_err: ", e)

    def generate_given_data_for_test(self, scale):
        try:
            # pass
            # self.pos_err = [np.random.random(pos_bound), np.random.random(pos_bound),np.random.random(pos_bound)]
            self.pos_err = np.array([0.5, 4.654, 0.5]) * scale
            self.spd_start = np.array([2.5, 0.5, 0.05456]) * scale
            self.acc_start = np.array([0.545, 6.54, 0.55]) * scale

            self.spd_end = np.array([3.5, 1.56789, 0.088]) * scale
            self.acc_end = np.array([1.46574, 4.24, -2.45]) * scale

            # self.cross_spd = 0.29051678
            self.cross_spd = 2.8
            # self.cross_spd = np.random.uniform()* (cross_spd_bound[1] - cross_spd_bound[0]) + cross_spd_bound[0]
            # print("============================")
            # print("cross_spd = ", self.cross_spd)
            # print("Pos_err = ", self.pos_err)
            # print("spd_start = ", self.spd_start)
            # print("acc_start = ", self.acc_start)
            #
            # print("spd_end = ", self.spd_end)
            # print("acc_end = ", self.acc_end)

            # self.generate_trajectory(self.pos_err, self.spd_start, self.acc_start,
            #                          self.spd_end, self.acc_end, self.cross_spd)
            self.generate_trajectory_cross_time(self.pos_err, self.spd_start, self.acc_start,
                                                self.spd_end, self.acc_end, self.cross_spd)
        except Exception as e:
            print("generate_random_trajectory_err: ", e)

    def export_data(self, sample_size=100, if_random = 0):
        try:
            self.input_data = np.zeros((sample_size, 17))  # t, cross_spd, pos_err, spd_start, acc_start, spd_end, acc_end
            self.output_data = np.zeros((sample_size, 9))  # pos(t), spd(t), acc(t)
            if(if_random):
                sample_t = np.random.uniform(size=sample_size) * self.Tf
                sample_t = np.sort(sample_t)
            else:
                sample_t = np.array(list(range(0, sample_size)))*self.Tf/(sample_size-1)
            for idx, t in enumerate(sample_t):
                pos_t = self.rapid_traj.get_position(t)
                spd_t = self.rapid_traj.get_velocity(t)
                acc_t = self.rapid_traj.get_acceleration(t) - self.gravity

                self.input_data[idx, 0] = t  # input time
                self.input_data[idx, 1] = self.cross_spd  # cross speed

                self.input_data[idx, 2] = float(self.pos_err[0])  # pos_start_3X1
                self.input_data[idx, 3] = float(self.pos_err[1])
                self.input_data[idx, 4] = float(self.pos_err[2])

                self.input_data[idx, 5] = float(self.spd_start[0])  # spd_start_3X1
                self.input_data[idx, 6] = float(self.spd_start[1])
                self.input_data[idx, 7] = float(self.spd_start[2])

                self.input_data[idx, 8] = float(self.acc_start[0])  # acc_start_3X1
                self.input_data[idx, 9] = float(self.acc_start[1])
                self.input_data[idx, 10] = float(self.acc_start[2])

                self.input_data[idx, 11] = float(self.spd_end[0])  # ...
                self.input_data[idx, 12] = float(self.spd_end[1])
                self.input_data[idx, 13] = float(self.spd_end[2])

                self.input_data[idx, 14] = float(self.acc_end[0])  # ...
                self.input_data[idx, 15] = float(self.acc_end[1])
                self.input_data[idx, 16] = float(self.acc_end[2])

                self.output_data[idx, 0] = float(pos_t[0])
                self.output_data[idx, 1] = float(pos_t[1])
                self.output_data[idx, 2] = float(pos_t[2])

                self.output_data[idx, 3] = float(spd_t[0])
                self.output_data[idx, 4] = float(spd_t[1])
                self.output_data[idx, 5] = float(spd_t[2])

                self.output_data[idx, 6] = float(acc_t[0])
                self.output_data[idx, 7] = float(acc_t[1])
                # self.output_data[idx, 8] = float(acc_t[2])-9.80
                self.output_data[idx, 8] = float(acc_t[2])

            return self.input_data, self.output_data
            # print("t = ", sample_t)
        except Exception as e:
            print("export_data erros = ", e)

    def plot_data(self, window_name='visualize'):
        plt.figure(window_name)
        title = (' Cross_spd = %.2f \n' % (self.input_data[0, 1]) + ' pos_err = ' + str(self.input_data[0, (2, 3, 4)].tolist()) + '\n'
                 + ' spd_start = ' + str(self.input_data[0, (5, 6, 7)].tolist()) + '\n'
                 + ' acc_start = ' + str(self.input_data[0, (8, 9, 10)].tolist()) + '\n'
                 + ' spd_end = ' + str(self.input_data[0, (11, 12, 13)].tolist()) + '\n'
                 + ' acc_end = ' + str(self.input_data[0, (14, 15, 16)].tolist()) + '\n'
                 )
        # print(title)
        # plt.suptitle( title )
        plt.subplot(3, 1, 1)
        plt.plot(self.input_data[:, 0], self.output_data[:, 0], 'r-', label='pos_err_x')
        plt.plot(self.input_data[:, 0], self.output_data[:, 1], 'k-', label='pos_err_y')
        plt.plot(self.input_data[:, 0], self.output_data[:, 2], 'b-', label='pos_err_z')
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.input_data[:, 0], self.output_data[:, 3], 'r-', label='spd_x')
        plt.plot(self.input_data[:, 0], self.output_data[:, 4], 'k-', label='spd_y')
        plt.plot(self.input_data[:, 0], self.output_data[:, 5], 'b-', label='spd_z')
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.input_data[:, 0], self.output_data[:, 6], 'r-', label='acc_x')
        plt.plot(self.input_data[:, 0], self.output_data[:, 7], 'k-', label='acc_y')
        plt.plot(self.input_data[:, 0], self.output_data[:, 8], 'b-', label='acc_z')
        plt.grid()
        plt.legend()

        plt.pause(0.1)

    def plot_data_dash(self, window_name='visualize'):
        plt.figure(window_name)
        title = (' Cross_spd = %.2f \n' % (self.input_data[0, 1]) + ' pos_err = ' + str(self.input_data[0, (2, 3, 4)].tolist()) + '\n'
                 + ' spd_start = ' + str(self.input_data[0, (5, 6, 7)].tolist()) + '\n'
                 + ' acc_start = ' + str(self.input_data[0, (8, 9, 10)].tolist()) + '\n'
                 + ' spd_end = ' + str(self.input_data[0, (11, 12, 13)].tolist()) + '\n'
                 + ' acc_end = ' + str(self.input_data[0, (14, 15, 16)].tolist()) + '\n'
                 )
        # print(title)
        # plt.suptitle( title )
        plt.subplot(3, 1, 1)
        plt.plot(self.input_data[:, 0], self.output_data[:, 0], 'r--', label='vali_pos_err_x')
        plt.plot(self.input_data[:, 0], self.output_data[:, 1], 'k--', label='vali_pos_err_y')
        plt.plot(self.input_data[:, 0], self.output_data[:, 2], 'b--', label='vali_pos_err_z')
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.input_data[:, 0], self.output_data[:, 3], 'r--', label='vali_spd_x')
        plt.plot(self.input_data[:, 0], self.output_data[:, 4], 'k--', label='vali_spd_y')
        plt.plot(self.input_data[:, 0], self.output_data[:, 5], 'b--', label='vali_spd_z')
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.input_data[:, 0], self.output_data[:, 6], 'r--', label='vali_acc_x')
        plt.plot(self.input_data[:, 0], self.output_data[:, 7], 'k--', label='vali_acc_y')
        plt.plot(self.input_data[:, 0], self.output_data[:, 8], 'b--', label='vali_acc_z')
        plt.grid()
        plt.legend()

        # plt.pause(0.1)

    def save_data_to_file(self, file_name):
        package_data = {"input_data": self.input_data}
        package_data.update({"output_data": self.output_data})
        file = open(file_name, 'wb')
        pickle.dump(package_data, file)
        file.close()

    def load_from_file(self, file_name):
        file = open(file_name, "rb")
        data = pickle.load(file)
        try:
            self.input_data = data["input_data"]
            self.output_data = data["output_data"]
        except:
            self.input_data = data["input_data_for_test"]
            self.output_data = data["output_data_for_test"]

        file.close()
        return self.input_data, self.output_data

    def load_from_file_vector(self, logger_name_vec, if_limit = 1):
        self.input_data_all = np.zeros((1000 * len(logger_name_vec), 17))
        self.output_data_all = np.zeros((1000 * len(logger_name_vec), 9))
        idx = 0
        for logger_name in logger_name_vec:
            # if (self.input_data_all is None):
            #     self.input_data_all, self.output_data_all = self.load_from_file(logger_name)
            # else:
            #     input_data_single_frame, output_data_single_frame = self.load_from_file(logger_name_vec[idx])
            #     self.input_data_all = np.concatenate((self.input_data_all, input_data_single_frame), axis=0)
            #     self.output_data_all = np.concatenate((self.output_data_all, output_data_single_frame), axis=0)
            file_input_data, file_output_data = self.load_from_file(logger_name)


            if(if_limit):
                if (np.max(np.abs(file_output_data[:, range(0, 3)])) > 30):
                    # print("pos error")
                    continue

                if (np.max(np.abs(file_output_data[:, range(3, 6)])) > 20):
                    # print("spd error")
                    continue

                if (np.max(np.abs(file_output_data[:, range(6, 8)])) > 10  or
                    np.max(np.abs(file_output_data[:, range(8, 9)])) > 30):
                    # print("acc error")
                    continue

                if(np.linalg.norm( file_input_data[0,range(2,5)]) < 0.6):
                    print('len error')
                    continue

            self.input_data_all[range(idx * 1000, (idx + 1) * 1000), :] = file_input_data
            self.output_data_all[range(idx * 1000, (idx + 1) * 1000), :] = file_output_data
            idx = idx + 1
            if (idx % 100 == 0):
                pass
                # print("idx = ", idx)
        print("data_avail_rate = %.3f" % (idx / len(logger_name_vec)))
        self.input_data_all = self.input_data_all[range(0, idx * 1000), :]
        self.output_data_all = self.output_data_all[range(0, idx * 1000), :]

        self.input_data = self.input_data_all[range(0, idx * 1000), :]
        self.output_data = self.output_data_all[range(0, idx * 1000), :]
        return self.input_data_all, self.output_data_all



def load_random_data(save_dir, file_size):
    sample_set = np.random.choice(range(0, file_size), file_size, replace=False)
    sample_set = np.sort(sample_set)
    # sample_set = range(file_size)
    # print(random_set)
    file_name_vec = []
    for idx in sample_set:
        name = "%s\\traj_%d.pkl" % (save_dir, idx)
        file_name_vec.append(name)

    rapid_trajectory = Rapid_trajectory_generator()
    # t_start = cv2.getTickCount()
    in_data, out_data = rapid_trajectory.load_from_file_vector(file_name_vec)
    # print("cost time  = %.2f " % ((cv2.getTickCount() - t_start) * 1000.0 / cv2.getTickFrequency()))
    return in_data, out_data



def traj_augment(input_data, output_data, if_sub_g = 0, traj_size = 1000):

    # Input[17]: time_1, cross_spd_1, pos_error_3, spd_start_3, acc_start_3, spd_end_3, acc_end_3
    #                 0,           1, 2         4, 5         7, 8        10, 11     13, 14     16
    # Output[9]: tar_pos_3, tar_spd_3, tar_acc_3

    # import copy
    t_start = cv2.getTickCount()
    if_regenerate = 1
    if_debug = 0
    if_symetry = 0
    if_reverse = 0
    if_pos_scale = 0
    if_time_scale = 1
    length = input_data.shape[0]
    traj_count = int(length / traj_size)
    # print("traj_augment, Length = ", length, " traj size =  ", traj_size, " count = ", traj_count)
    aug_data_input = copy.deepcopy(input_data)
    aug_data_output = copy.deepcopy(output_data)

    if(if_sub_g):
        # add first, subtract all
        aug_data_output[:,8] = aug_data_output[:,8] + 9.8

    rapid_trajectory = Rapid_trajectory_generator()
    # symmetry
    if(if_symetry):
        enh_input_data = copy.deepcopy(input_data)
        enh_output_data = copy.deepcopy(output_data)
        enh_input_data[:, range(2, 17)]  = enh_input_data[:, range(2, 17)]*-1.0
        # enh_output_data = enh_output_data*-1.0
        for traj_idx in range(traj_count):
            enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), :] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), :] * -1.0
            if(if_sub_g):
                enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), 8] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), 8]+9.8
            # for idx in range(traj_size):
            # # #     print(traj_idx*traj_size +idx ,' <--> ', (traj_idx+1)*traj_size- 1 - idx )
            #     enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size),:] = enh_output_data[range(traj_idx*traj_size, (traj_idx+1)*traj_size),:]*-1.0
            if(if_regenerate):
            # enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] = np.flip(enh_output_data[ range(traj_idx*traj_size, (traj_idx+1)*traj_size) ], axis=0)
                rapid_trajectory.generate_trajectory_np_array( enh_input_data[traj_idx*traj_size + 0] )
                print("sym compare = ", np.mean(np.abs(enh_output_data[range(traj_idx*traj_size, (traj_idx+1)*traj_size)] -rapid_trajectory.output_data )))
                # enh_input_data[range(traj_idx*traj_size, (traj_idx+1)*traj_size)] = rapid_trajectory.input_data
                # enh_output_data[range(traj_idx*traj_size, (traj_idx+1)*traj_size)] = rapid_trajectory.output_data

        aug_data_input = np.concatenate([aug_data_input, enh_input_data] , axis=0)
        aug_data_output = np.concatenate([aug_data_output, enh_output_data] , axis=0)

        if(if_debug):
            rapid_trajectory.input_data = enh_input_data
            rapid_trajectory.output_data = enh_output_data
            rapid_trajectory.plot_data()
            rapid_trajectory.input_data = input_data
            rapid_trajectory.output_data = output_data
            rapid_trajectory.plot_data_dash()

    if(if_reverse):
        enh_input_data = copy.deepcopy(input_data)
        enh_output_data = copy.deepcopy(output_data)

        enh_input_data[:, range(2,5)] =   copy.deepcopy(input_data[:, range(2,5)])*-1.0 # Pos error inverse
        enh_input_data[:, range(5,8)] =  -copy.deepcopy(input_data[:, range(11,14)])    # Swap start and end spd
        enh_input_data[:, range(8,11)] =  -copy.deepcopy(input_data[:, range(14,17)])   # Swap start and end acc
        enh_input_data[:, range(11,14)] =  -copy.deepcopy(input_data[:, range(5,8)])    # Swap start and end spd
        enh_input_data[:, range(14,17)] =  -copy.deepcopy(input_data[:, range(8,11)])   # Swap start and end acc
        for traj_idx in range(traj_count):
            tf = np.linalg.norm(enh_input_data[0, range(2, 5)]) / enh_input_data[:, 1]
            # print("tf = ", tf)

            for idx in range(traj_size):
                swap_idx =  (traj_idx+1) * traj_size - idx -1
                print(traj_idx*traj_size +idx ,' <--> ', swap_idx)
                enh_output_data[traj_idx * traj_size + idx, range(0, 3)] = copy.deepcopy(output_data[ swap_idx , range(0, 3)] + enh_input_data[0, range(2,5)])
                enh_output_data[traj_idx * traj_size + idx, range(3, 9)] = copy.deepcopy(output_data[ swap_idx , range(3, 9)]) * -1.0

                # enh_output_data[ traj_idx*traj_size +idx ] =  copy.deepcopy( output_data[ (traj_idx+1)*traj_size- 1 - idx ])*-1

            if (if_sub_g):
                enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), 8] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), 8] + 9.8

            if (if_regenerate):
                # enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] = np.flip(enh_output_data[ range(traj_idx*traj_size, (traj_idx+1)*traj_size) ], axis=0)
                rapid_trajectory.generate_trajectory_np_array(enh_input_data[traj_idx * traj_size + 1])
                # rapid_trajectory.export_data(1000, if_random=0)
                print("sym compare = ", np.mean(np.abs(enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] - rapid_trajectory.output_data)))
                # print("time rev compare = ", np.sum(np.abs(enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] - rapid_trajectory.output_data)))
                # print(enh_output_data.shape, rapid_trajectory.output_data.shape)
                # print(enh_output_data[ range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][ :, range(0,3) ])
                print("time rev compare = ", np.mean(np.abs( enh_output_data[ range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:,range(3,6)] -
                                                           rapid_trajectory.output_data[:, range(3,6)] ) ) )
                # enh_input_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] = rapid_trajectory.input_data
                # enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] = rapid_trajectory.output_data
                # enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] - rapid_trajectory.output_data

        aug_data_input = np.concatenate([aug_data_input, enh_input_data], axis=0)
        aug_data_output = np.concatenate([aug_data_output, enh_output_data], axis=0)

        if(if_debug):
            # print()
            plt.figure('t')
            plt.plot(enh_input_data[:,0], 'r')
            plt.plot(input_data[:,0], 'g')
            rapid_trajectory.input_data =  rapid_trajectory.input_data
            rapid_trajectory.output_data =  rapid_trajectory.output_data
            rapid_trajectory.plot_data_dash("compare")
            rapid_trajectory.input_data = enh_input_data
            rapid_trajectory.output_data = rapid_trajectory.output_data - enh_output_data
            rapid_trajectory.plot_data("compare")
    if(if_pos_scale):
        # small 10 time
        scale_para = traj_index = np.random.uniform(low=1, high=5.0, size=1)[0]
        # print("Pos scale  = ", scale_para)
        scale_factor = [scale_para, 1.0/scale_para]
        for scale in scale_factor:
            enh_input_data = copy.deepcopy(input_data)
            enh_output_data = copy.deepcopy(output_data)
            enh_input_data[:, range(1, 17)] = enh_input_data[:, range(1, 17)] * scale

            for traj_idx in range(traj_count):
                enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), :] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), :] * scale
                if (if_sub_g):
                    enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), 8] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), 8] + 9.8
                if(if_regenerate):
                    rapid_trajectory.generate_trajectory_np_array( enh_input_data[traj_idx*traj_size + 1] )
                    print("pose compare = ", np.mean(np.abs(enh_output_data[range(traj_idx*traj_size, (traj_idx+1)*traj_size)] -rapid_trajectory.output_data )))
                    enh_input_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] = rapid_trajectory.input_data
                    enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] = rapid_trajectory.output_data

            aug_data_input = np.concatenate([aug_data_input, enh_input_data], axis=0)
            aug_data_output = np.concatenate([aug_data_output, enh_output_data], axis=0)

    if(if_time_scale):
        # small 10 time
        # scale_para = traj_index = np.random.uniform(low=1, high=5.0, size=1)[0]
        scale_para = 2
        # print("time scale  = ", scale_para)
        scale_factor = [scale_para, 1.0/scale_para]
        for scale in scale_factor:
            enh_input_data = copy.deepcopy(input_data)
            enh_output_data = copy.deepcopy(output_data)*1

            enh_input_data[:, 1] = enh_input_data[:, 1] * scale
            enh_input_data[:, 0] = enh_input_data[:, 0] / scale
            for traj_idx in range(traj_count):
                # enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][ range(0,3)] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][ range(0,3)]
                # enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][ range(0,3)] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][ range(0,3)]
                print(enh_output_data.shape)
                print(enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:,range(0,3)].shape)
                for idx in range(traj_size):
                    enh_output_data[traj_idx * traj_size+idx, range(0, 3)] = enh_output_data[traj_idx * traj_size+idx, range(0, 3)] /scale
                    enh_output_data[traj_idx * traj_size+idx, range(3, 6)] = enh_output_data[traj_idx * traj_size+idx, range(3, 6)] * 1.0
                    enh_output_data[traj_idx * traj_size+idx, range(6, 9)] = enh_output_data[traj_idx * traj_size+idx, range(6, 9)] * scale

                # enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:,range(0,3)] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:,range(0,3)]*0
                # enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:,range(3,6)] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:,range(3,6)]*scale
                # enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:,range(6,9)] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:,range(6,9)]*scale*scale

                if (if_sub_g):
                    enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), 8] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size), 8] + 9.8

                if (if_regenerate):
                    rapid_trajectory.generate_trajectory_np_array( enh_input_data[traj_idx*traj_size + 1] )
                    print("time compare = ", np.sum(np.abs(enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] - rapid_trajectory.output_data)))
                    enh_input_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] = rapid_trajectory.input_data
                    enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] = rapid_trajectory.output_data
                # print("Scale  = ", scale)
                # print("Pos norm_max = ",  np.linalg.norm( rapid_trajectory.output_data[:,range(0,3)] ))
                # print("Spd norm_max = ", np.linalg.norm(rapid_trajectory.output_data[:, range(3, 6)]))
                # rapid_trajectory.input_data[:, 8] = rapid_trajectory.output_data[:, 8] - 9.8
                # print("Acc norm_max = ", np.linalg.norm(rapid_trajectory.output_data[:, range(6, 9)]))
                aug_data_input = np.concatenate([aug_data_input, enh_input_data], axis=0)
                aug_data_output = np.concatenate([aug_data_output, enh_output_data], axis=0)


    if(if_debug):
        plt.show()

    if(if_sub_g):
        # add first, subtract all
        aug_data_output[:,8] = aug_data_output[:,8] - 9.8
    # print('Data argument cost time = ',  (cv2.getTickCount() - t_start)/cv2.getTickFrequency())
    return aug_data_input, aug_data_output

    # Time argument
    pass

def generate_crossgap_example(dir):
    generate_list = []

    generate_list.append(np.array([0.0, 0.605, -1.178, 0.788, 0.779, 0.363, -0.041, 0.211, -0.438, 0.051, -0.234, 7.0, 0.0, 0.0, 0.0, 4.244, -2.45]))
    # generate_list.append(np.array([0.0, 4.0, 19.5, -10.043301270189222, -4.975, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.4999999807142856, 0.0, 0.0, 0.0, 4.243524431778378, -2.450000080999998]));
    # # generate_list.append(np.array([])); # Copy this
    # generate_list.append(np.array([0.0, 0.758, -0.705, -1.132, 1.24, 0.332, 0.043, 0.167, -0.454, -0.074, -0.187, 7.0, 0.0, 0.0, 0.0, 4.244, -2.45]))   # Copy this
    # generate_list.append(np.array([0.0, 5.0, 19.346, -9.911, -4.644, -0.177, 0.157, 0.381, 0.156, -0.214, -0.422, 3.5, 0.0, 0.0, 0.0, 4.244, -2.45]))
    # generate_list.append(np.array([0.0, 0.472, -1.073, -0.149, 0.387, -0.09, 0.045, -0.226, 0.375, 0.211, 0.245, 7.0, 0.0, 0.0, 0.0, 4.244, -2.45]))    # Copy this
    # generate_list.append(np.array([]))
    # generate_list.append(np.array([]));
    # generate_list.append(np.array([]));
    idx= 0
    if_sub_g = 1
    for traj in generate_list:
        file_name = "%s/traj_example_%d.pkl"%(dir,idx)
        rapid_trajectory = Rapid_trajectory_generator()
        rapid_trajectory.generate_trajectory_np_array(traj)

        rapid_trajectory.export_data(1000, if_random=0)
        if (if_sub_g):
            rapid_trajectory.output_data[:, 8] = rapid_trajectory.output_data[:, 8] - 9.8
        rapid_trajectory.input_data, rapid_trajectory.output_data = traj_augment( rapid_trajectory.input_data, rapid_trajectory.output_data, if_sub_g=1 )
        # rapid_trajectory.input_data, rapid_trajectory.output_data = traj_augment( rapid_trajectory.input_data, rapid_trajectory.output_data )
        # rapid_trajectory.input_data, rapid_trajectory.output_data = traj_augment( rapid_trajectory.input_data, rapid_trajectory.output_data )
        # rapid_trajectory.save_data_to_file( file_name)
        idx=idx+1
        rapid_trajectory.plot_data()
        plt.show()
        break

    print(generate_list)
    print(len(generate_list))



if __name__ == "__main__":
    print("hello, this is rapid_trajectory_generator.")
    # save_dir = "%s/planner_given"%(os.getenv('DL_data'))
    save_dir = "%s/planner"%(os.getenv('DL_data'))
    # save_dir = "%s/planner_no_g_second_v" % (os.getenv('DL_data'))
    print("Save dir = ", save_dir)
    # save_dir = "I:/data/planner_short/"
    generate_crossgap_example(save_dir)
    exit()
    try:
        os.mkdir(save_dir)
    except Exception as e:
        print(e)
    rapid_trajectory = Rapid_trajectory_generator()

    bias = 15000
    sample_size = 20000
    t_start = cv2.getTickCount()
    load_data = 1
    if (load_data):
        input_data, output_data = load_random_data(save_dir, sample_size)
        # input_data_for_test, output_data_for_test = rapid_trajectory.load_from_file("%s/batch_%d.pkl"%(save_dir, sample_size))
        print("cost time  = %.2f " % ((cv2.getTickCount() - t_start) * 1000.0 / cv2.getTickFrequency()))

        rapid_trajectory.input_data = input_data
        rapid_trajectory.output_data = output_data
        # rapid_trajectory.save_data_to_file("%s/batch_remove_outlinear_%d.pkl" % (save_dir, sample_size))
        rapid_trajectory.save_data_to_file("%s/batch_2_%d.pkl" % (save_dir, sample_size))
        # plt.plot(rapid_trajectory.input_data[:,1])
        plt.plot(np.linalg.norm( rapid_trajectory.input_data[:,range(2,5)], axis = 1 ,keepdims=1) )
        print("Min time =  ", )
        # rapid_trajectory.plot_data()
        plt.show()
        exit()

    generata_size = 5000
    for k in range(generata_size):
        if (k % 100 == 0):
            print(k)
        file_name = "%s/traj_%d.pkl" % (save_dir, k + bias)
        # rapid_trajectory.generate_random_trajectory([-1, 1], [-5, 5], [-5, 5], [1, 7])
        # rapid_trajectory.generate_random_trajectory_cross_time([-5, 5], [-5, 5], [-5, 5], [1, 7])
        rapid_trajectory.generate_given_data([-1, 1], [-10, 10], [-10, 10], [0, 0.5])
        # rapid_trajectory.generate_given_data()
        # scale = (k + 1) / generata_size

        # rapid_trajectory.generate_given_data_for_test(scale)
        rapid_trajectory.export_data(1000)

        # rapid_trajectory.output_data = rapid_trajectory.output_data / scale
        # rapid_trajectory.plot_data()
        rapid_trajectory.save_data_to_file(file_name)
    print("Finish")
    plt.show()

    # sample_size = generata_size
    # if(1):
    #     input_data, output_data = load_random_data(save_dir , sample_size  )
    #     # input_data_for_test, output_data_for_test = rapid_trajectory.load_from_file("%s/batch_%d.pkl"%(save_dir, sample_size))
    #     print("cost time  = %.2f " % ((cv2.getTickCount() - t_start) * 1000.0 / cv2.getTickFrequency()))
    # 
    #     rapid_trajectory.input_data = input_data
    #     rapid_trajectory.output_data = output_data
    #     rapid_trajectory.save_data_to_file("%s/batch_remove_outlinear_%d.pkl"%(save_dir, sample_size))
    #     rapid_trajectory.plot_data()
    #     plt.show()
    #     exit()
