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
import sys, os

sys.path.append("..")  # Adds higher directory to python modules path.

from query_data.tools.plane_logger import plane_logger
from trajectory_data_loder import Trajectory_data_loader

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

    trajectory_data_loader = Trajectory_data_loader()
    input_data, output_data = trajectory_data_loader.load_data_from_logger_vec(logger_name_vec)

    print("All data_size = %d"%(input_data.shape[0]))

    torch_input_data = torch.from_numpy(input_data).float().cuda()
    torch_output_data = torch.from_numpy(output_data).float().cuda()

    n_hidden_per_layer = 20
    net = torch.nn.Sequential(
        torch.nn.Linear(9, n_hidden_per_layer),
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
        torch.nn.Linear(n_hidden_per_layer, 4)).cuda()  # method 2

    print(net)
    learn_rate = 0.5
    optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)
    # optimizer = torch.optim.Adam(net.parameters(), lr= learn_rate)
    loss_func = torch.nn.MSELoss().cuda()
    for t in range(1000000):
        prediction = net(torch_input_data)
        loss = loss_func(prediction, torch_output_data).cuda()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (t % 5000 == 0):
            learn_rate = learn_rate * 0.95
            optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate / 2)
        if (t % 10000 == 0 ):
            torch.save(net, "demo_network_final.pkl")

        if t % 100 == 0:
            print('t =', t, " loss", loss)
        if t % 1000 == 0:
            torch.save(net, "%s/demo_network_%d.pkl" % (model_save_dir, t))
    torch.save(net, "demo_network_final.pkl")

    # plt.show()
