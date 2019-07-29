from __future__ import print_function
from torch import *
import torch.nn
import torch.nn.functional
import torch.optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np
import cv2 as cv2
import sys, os, math
import torch
import pickle

sys.path.append("..")  # Adds higher directory to python modules path.

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Disable GPU using
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from query_data.tools.plane_logger import plane_logger

def load_pid_data():
    data_size = int(6e6)
    DATA_DIR = "%s/pid_data" % (os.getenv('DL_data'))
    print("Data save dir =  ", DATA_DIR)
    # pkl_file_name = "%s/pid_data_%s.pkl" % (DATA_DIR, str(data_size))
    if(1):
        pkl_file_name = "%s/pid_data_big_%s.pkl" % (DATA_DIR, str(int(6e6)))
        data_dict_big_range = pickle.load(open(pkl_file_name, 'rb'))

        pkl_file_name = "%s/pid_data_%s.pkl"%(DATA_DIR, str(int(1e6)))
        data_dict_short_range = pickle.load(open(pkl_file_name, 'rb'))
        # in_data = data_dict_short_range['in_data']
        # out_data = data_dict_short_range['out_data']

        in_data = np.concatenate([data_dict_big_range['in_data'], data_dict_short_range['in_data']], axis= 0)
        out_data = np.concatenate([data_dict_big_range['out_data'], data_dict_short_range['out_data']], axis= 0)
    else:
        pkl_file_name = "%s/pid_data_lim_%s.pkl"%(DATA_DIR, str(data_size))
        data_dict_lim = pickle.load(open(pkl_file_name, 'rb'))
        in_data = data_dict_lim['in_data']
        out_data = data_dict_lim['out_data']
    print("Load_data size = ", in_data.shape[0])
    return in_data, out_data


class Self_def_loss(torch.nn.Module):
    def __init__(self, weight_vec):
        super(Self_def_loss, self).__init__()
        self.weight_vec = weight_vec
        return

    def rotation_error(self, net_roll, net_pitch, tar_roll, tar_pitch ):

        cx1 = torch.cos(net_roll)
        sx1 = torch.sin(net_roll)
        cy1 = torch.cos(net_pitch)
        sy1 = torch.sin(net_pitch)

        cx2 = torch.cos(tar_roll)
        sx2 = torch.sin(tar_roll)
        cy2 = torch.cos(tar_pitch)
        sy2 = torch.sin(tar_pitch)

        m00 = cy1*cy2 + sy1*sy2
        m11 = cx1*cx2 + cy1*cy2*sx1*sx2 + sx1*sx2*sy1*sy2
        m22 = sx1*sx2 + cx1*cx2*sy1*sy2 + cx1*cx2*cy1*cy2
        self.acos =  ( m00 + m11 + m22 - 1)/2.0 * 0.99999
        return torch.acos(self.acos)

    def forward(self, prediction, target):
        # print("Run forward")
        roll_err_vec = prediction[:, (0)] - target[:, (0)]
        pitch_err_vec = prediction[:, (1)] - target[:, (1)]
        thrust_err_vec = prediction[:, (2)] - target[:, (2)]
        # print(roll_err_vec)
        # print("----")
        # print(pitch_err_vec)

        # print(prediction.shape, " --- ", pos_err_vec.shape)
        # print(pos_err_vec)
        # print("pos_err_vec_norm = ", err_vec_withweight_norm)

        # err_vec_withweight_norm = torch.pow(roll_err_vec, 2)*self.weight_vec[0]+ torch.pow(pitch_err_vec, 2)*self.weight_vec[1] + torch.pow(thrust_err_vec, 2)*self.weight_vec[2]
        err_vec_withweight_norm = self.rotation_error(prediction[:, (0)], prediction[:, (1)], target[:, (0)], target[:, (1)]) * self.weight_vec[1] + torch.pow(thrust_err_vec, 2) * self.weight_vec[2]
        return err_vec_withweight_norm.mean()

if __name__ == "__main__":
    torch.manual_seed(1)  # reproducible

    logger_name_vec = []
    for idx in range(0,8):
        logger_name_vec.append("G:/data/narrow_gap/for_test_%d.txt" % (idx))

    model_save_dir = './model_temp_2'
    try:
        os.mkdir(model_save_dir)
    except Exception as e:
        print(e)

    input_data_for_train, output_data_for_train  =  load_pid_data()

    print("All data_size = %d" % (input_data_for_train.shape[0]))

    input_data_for_test = input_data_for_train[range(100), :]
    output_data_for_test = output_data_for_train[range(100), :]

    # BATCH_SIZE = int(1500 * 1.3 * 100)
    BATCH_SIZE = 100
    BATCH_COUNT = input_data_for_train.shape[0] / BATCH_SIZE

    torch_input_data = torch.from_numpy(input_data_for_train).float().cuda()
    torch_output_data = torch.from_numpy(output_data_for_train).float().cuda()

    n_hidden_per_layer = 40
    net = torch.nn.Sequential(
        torch.nn.Linear(12, n_hidden_per_layer),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        # torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        # torch.nn.ReLU(),
        # torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        # torch.nn.Sigmoid(),
        # torch.nn.Softplus(),
        # torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, 3)).cuda()  # method 2
    print(net)
    bias_t = 2100000 # 1.0
    bias_t = 2230000
    bias_t = 620000 # before change loss
    bias_t = 3700000 # Loss 2.4!!! successful train
    bias_t = 20710000
    # bias_t = 4680000

    if(bias_t !=0):
        net = torch.load("%s/demo_network_%d.pkl" % (model_save_dir, bias_t))
    learn_rate = 1e-5
    # optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)
    optimizer = torch.optim.Adam(net.parameters(), lr= learn_rate)
    loss_func = Self_def_loss([57.3,57.3, 2])
    # loss_func = torch.nn.MSELoss().cuda()

    save_index = 10000
    t = bias_t
    print("t = ", t)
    BATCH_SIZE = 100
    BATCH_COUNT = input_data_for_train.shape[0] / BATCH_SIZE
    for epoch in range(100000000000000000):

        learn_rate = learn_rate * 0.995
        BATCH_SIZE = int(BATCH_SIZE+1)
        BATCH_COUNT = input_data_for_train.shape[0] / BATCH_SIZE
        print("Lr = ", learn_rate, ", batchsize = ", BATCH_SIZE)
        # optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate / 2)
        optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)
        # torch.save(net, "demo_network_final.pkl")

        for loop_1 in range(0, math.ceil(BATCH_COUNT/5)):

            traj_index = np.random.randint(low=0, high=BATCH_COUNT - 1, size=1)[0]
            # print("traj_index = ", traj_index)
            np_data_x = input_data_for_train[range(traj_index * BATCH_SIZE, min(len(input_data_for_train) - 1, (traj_index + 1) * BATCH_SIZE)), :]
            np_data_y = output_data_for_train[range(traj_index * BATCH_SIZE, min(len(input_data_for_train) - 1, (traj_index + 1) * BATCH_SIZE)), :]

            prediction = net(torch.from_numpy(np_data_x).float().cuda())
            loss = loss_func(prediction, torch.from_numpy(np_data_y).float().cuda()).cuda()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 1000 == 0:
                print('t =', t, " loss", loss)

            if (t % save_index == 0 and t !=bias_t):
                try:
                    torch.save(net, "%s/demo_network_%d.pkl" % (model_save_dir, t))
                    print("Save net successful!!! %d %%" %(loop_1*100/math.ceil(BATCH_COUNT)) )
                except Exception as e:
                    pass
            t = t + 1
