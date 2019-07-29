import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys, os
import multiprocessing as mp
import torch

sys.path.append("..")  # Adds higher directory to python modules path.

from query_data.tools.plane_logger import plane_logger



def thread_data_logger_kernel():
    pass

class Trajectory_data_loader():

    def __init__(self):
        self.logger = plane_logger()
        self.input_dimension = 9  # [ pos_err_3, spd_err_3, acc_vec_3, ]
        self.output_dimension = 4  # [r,p,y,throttle]
        self.input_data_frame_vec = []
        self.output_data_frame_vec = []
        self.input_data_all = None
        self.output_data_all = None

    @staticmethod
    def extract_data(data):
        data_size = np.array(data["time_stamp"]).shape[0]
        print("Data size = %d" % (data_size))
        input_dimension = 9
        output_dimension = 4
        input_data = np.zeros((data_size, input_dimension))
        output_data = np.zeros((data_size, output_dimension))

        # load data
        for idx in range(data_size):
            # Input data
            pos_err_3 = data["target_pos"][idx] - data["current_pos"][idx]
            spd_err_3 = data["target_spd"][idx] - data["current_spd"][idx]
            # acc_err_3 = data["target_acc"][idx] - data["current_acc"][idx]    # Fucking bug.
            acc_err_3 = data["target_acc"][idx]

            # input_data_for_test[idx, 0] = float(pos_err_3[0])
            # input_data_for_test[idx, 1] = float(pos_err_3[1])
            # input_data_for_test[idx, 2] = float(pos_err_3[2])
            #
            # input_data_for_test[idx, 3] = float(spd_err_3[0])
            # input_data_for_test[idx, 4] = float(spd_err_3[1])
            # input_data_for_test[idx, 5] = float(spd_err_3[2])
            #
            # input_data_for_test[idx, 6] = float(acc_err_3[0])
            # input_data_for_test[idx, 7] = float(acc_err_3[1])
            # input_data_for_test[idx, 8] = float(acc_err_3[2])

            input_data[idx,range(0,3)] = pos_err_3.ravel()
            input_data[idx,range(3,6)] = spd_err_3.ravel()
            input_data[idx,range(6,9)] = acc_err_3.ravel()

            # output data
            output_data[idx, 0] = float(data["input_roll"][idx])
            output_data[idx, 1] = float(data["input_pitch"][idx])
            output_data[idx, 2] = float(data["input_yaw"][idx])
            output_data[idx, 3] = float(data["input_throttle"][idx])

            np.set_printoptions(precision=2)
            # print(input_data_for_test[idx,:], output_data_for_test[idx,:])
        return input_data, output_data

    @staticmethod
    def single_thread_data_logger_kernel(logger_name):
        print("load data form %s " % (logger_name))
        logger =  plane_logger()
        logger.load(logger_name)
        data = logger.data
        print(logger_name, " finish load")
        return  Trajectory_data_loader.extract_data(data)

    @staticmethod
    def mutilple_thread_data_logger_kernel(logger_name, q):
        input_data, output_data = Trajectory_data_loader.single_thread_data_logger_kernel(logger_name)
        # print("Before add queue")
        q.put((input_data, output_data))
        # print("After add queue")
        return

    def load_data_from_logger(self, logger_name):
        input_data = None
        output_data = None
        if(1):  #Both method is available
            self.logger.load(logger_name)
            data = self.logger.data
            print("load data form %s " % (logger_name))
            input_data, output_data = self.extract_data(data)
        else:
            input_data, output_data = self.single_thread_data_logger_kernel(logger_name)
        self.input_data_frame_vec.append(input_data)
        self.output_data_frame_vec.append(output_data)
        return input_data, output_data

    def load_data_from_logger_vec_single_thread(self, logger_name_vec):
        print("=== Load data from single thread ===")
        for idx, logger_name in enumerate(logger_name_vec):
            if (self.input_data_all is None):
                self.input_data_all, self.output_data_all = self.load_data_from_logger(logger_name_vec[idx])
            else:
                input_data_single_frame, output_data_single_frame = self.load_data_from_logger(logger_name_vec[idx])
                self.input_data_all = np.concatenate((self.input_data_all, input_data_single_frame), axis=0)
                self.output_data_all = np.concatenate((self.output_data_all, output_data_single_frame), axis=0)
        print(logger_name, " finish load")
        return self.input_data_all, self.output_data_all

    def load_data_from_logger_mutilple_thread(self, logger_name_vec):
        print("=== Load data from multiple thread ===")
        thread_vec = []
        q = mp.Queue()
        for logger_name in logger_name_vec:
            thread = mp.Process(target=Trajectory_data_loader.mutilple_thread_data_logger_kernel, args=(logger_name,q))
            thread.start()
            thread_vec.append(thread)

        for thd in thread_vec:
            # print("-->1")
            if (self.input_data_all is None):
                self.input_data_all, self.output_data_all = q.get()
                self.input_data_frame_vec.append(self.input_data_all)
                self.output_data_frame_vec.append(self.output_data_all)
            else:
                input_data_single_frame, output_data_single_frame = q.get()
                self.input_data_all = np.concatenate((self.input_data_all, input_data_single_frame), axis=0)
                self.output_data_all = np.concatenate((self.output_data_all, output_data_single_frame), axis=0)
                self.input_data_frame_vec.append(input_data_single_frame)
                self.output_data_frame_vec.append(output_data_single_frame)
            # thd.join()    # Some mystoryous hera, when you use queue, you will be block at XXX.join
            # print("-->2")
        print(self.input_data_all.shape)
        return self.input_data_all, self.output_data_all

    def load_data_from_logger_vec(self, logger_name_vec):
        if (1):
            return self.load_data_from_logger_vec_single_thread(logger_name_vec)
        else:
            return self.load_data_from_logger_mutilple_thread(logger_name_vec)

    def plot_data(self, idx):

        plt.figure("trajectory_visualize_%d" % (idx))
        plt.subplot(4, 1, 1)
        plt.plot(self.input_data_frame_vec[idx][:, 0], "r", label='err_pos_x')  # ,marker ="+")
        plt.plot(self.input_data_frame_vec[idx][:, 1], "k", label='err_pos_y')  # ,marker ="+")
        plt.plot(self.input_data_frame_vec[idx][:, 2], "b", label='err_pos_z')  # ,marker ="+")
        plt.legend(loc="best")

        plt.subplot(4, 1, 2)
        plt.plot(self.input_data_frame_vec[idx][:, 3], "r", label='err_spd_x')  # ,marker ="+")
        plt.plot(self.input_data_frame_vec[idx][:, 4], "k", label='err_spd_y')  # ,marker ="+")
        plt.plot(self.input_data_frame_vec[idx][:, 5], "b", label='err_spd_z')  # ,marker ="+")
        plt.legend(loc="best")

        plt.subplot(4, 1, 3)
        plt.plot(self.input_data_frame_vec[idx][:, 6], "r", label='err_acc_x')  # ,marker ="+")
        plt.plot(self.input_data_frame_vec[idx][:, 7], "k", label='err_acc_y')  # ,marker ="+")
        plt.plot(self.input_data_frame_vec[idx][:, 8], "b", label='err_acc_z')  # ,marker ="+")
        plt.legend(loc="best")

        plt.subplot(4, 1, 4)
        # plt.figure("trajectory_visualize_output")
        plt.plot(self.output_data_frame_vec[idx][:, 0], "r", label='input_roll')  # ,marker ="+")
        plt.plot(self.output_data_frame_vec[idx][:, 1], "k", label='input_pitch')  # ,marker ="+")
        # plt.plot(self.output_data_frame_vec[idx][:, 2], "b", label='input_yaw')  # ,marker ="+")
        plt.plot(self.output_data_frame_vec[idx][:, 3], "y", label='input_throttle')  # ,marker ="+")
        plt.legend(loc="best")

        plt.pause(0.1)

    def eval_net_rnn_nocuda(self, torch_net, idx):
        torch_net = torch_net.cpu()
        hidden_state = None
        torch_input_data = torch.from_numpy(self.input_data_frame_vec[idx][np.newaxis, :, :]).float()
        print(torch_input_data.shape)
        prediction, hidden_state = torch_net(torch_input_data, hidden_state)

        output_data = prediction.data.cpu().numpy()[0, :, :]
        self.eval_net(output_data, idx)

    def eval_net(self, output_data, idx=0):
        self.plot_data(idx)
        plt.figure("trajectory_visualize_%d" % (idx))

        plt.subplot(4, 1, 4)
        # plt.figure("trajectory_visualize_output")
        plt.plot(output_data[:, 0], "r--", label='rnn_roll')  # ,marker ="+")
        plt.plot(output_data[:, 1], "k--", label='rnn_pitch')  # ,marker ="+")
        # plt.plot(output_data[:, 2], "b--", label='rnn_yaw')  # ,marker ="+")
        plt.plot(output_data[:, 3], "y--", label='rnn_throttle')  # ,marker ="+")
        plt.legend(loc="best")
        plt.pause(0.1)

    def eval_net_rnn(self, torch_net, idx):
        # self.plot_data(idx)
        hidden_state = None
        torch_input_data = torch.from_numpy(self.input_data_frame_vec[idx][np.newaxis, :, :]).float().cuda()
        print(torch_input_data.shape)
        prediction, hidden_state = torch_net(torch_input_data, hidden_state)

        output_data = prediction.data.cpu().numpy()[0, :, :]
        self.eval_net(output_data, idx)
