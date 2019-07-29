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

from trajectory_rnn_net import Rnn_net
from trajectory_data_loder import Trajectory_data_loader

if __name__ == "__main__":

    trajectory_data_loader = Trajectory_data_loader()
    torch.manual_seed(1)  # reproducible

    logger_name_vec = []
    logger_name_vec.append("G:/data/narrow_gap/log_-10_10_10_4.0.txt")
    # for idx in range(0,8):
    for idx in range(0, 8):
        logger_name_vec.append("G:/data/narrow_gap/for_test_%d.txt" % (idx))

    model_save_dir = './model_temp_rnn_batch'
    try:
        os.mkdir(model_save_dir)
    except Exception as e:
        print(e)

    input_data, output_data = trajectory_data_loader.load_data_from_logger_vec(logger_name_vec)
    # rnn_net_for_eval = Rnn_net()
    # rnn_net_for_eval = torch.load("demo_network_final.pkl")
    # trajectory_data_loader.eval_net_rnn(rnn_net_for_eval, 0 )
    # trajectory_data_loader.plot_data(0)
    # plt.show()
    # exit()
    print("All data_size = %d"%(input_data.shape[0]))

    # torch_input_data = torch.from_numpy(input_data_for_test[np.newaxis,:,:]).float().cuda()
    # torch_output_data = torch.from_numpy(output_data_for_test[np.newaxis,:,:]).float().cuda()
    # hidden_state = None

    n_hidden_per_layer = 20
    net = Rnn_net().cuda()
    print(net)

    learn_rate = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)
    # optimizer = torch.optim.Adam(net.parameters(), lr= learn_rate)
    loss_func = torch.nn.MSELoss()

    max_time = 10000
    max_epoch  =10000
    batch_size = 200

    for epoch in range(max_epoch):

        shape = input_data.shape
        start_idx = np.random.randint(0, shape[0] - batch_size)
        if(batch_size > shape[0]):
            batch_size = shape[0] -1
        sample_size = (np.linspace(start_idx, start_idx + batch_size - 1, batch_size).astype(int))
        # print(start_idx)
        # print(sample_size)
        sample_input_data = input_data[sample_size]
        sample_output_data = output_data[sample_size]
        torch_input_data = torch.from_numpy(sample_input_data[np.newaxis, :, :]).float().cuda()
        torch_output_data = torch.from_numpy(sample_output_data[np.newaxis, :, :]).float().cuda()
        hidden_state = None


        for t in range(100):
            prediction, hidden_state = net(torch_input_data, hidden_state)
            loss = loss_func(prediction, torch_output_data)

            hidden_state = hidden_state.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (t % (max_time / 100) == 0):
                learn_rate = learn_rate * 1
                optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)
                # torch.save(net, "%s/demo_network_%d.pkl" % (model_save_dir, t))
            if (t % (max_time / 100) == 0):
                print("save net work to ./demo_network_final_rnn.pkl")
                # torch.save(net, "demo_network_final.pkl")

            # if (t % (max_time / 1) == 0):
            #     print('t =', t, "lr = %.2f"% (learn_rate) , " loss = ", loss )
            print('epoch = %d, '%(epoch) ,' t =', t, "lr = %.2f" % (learn_rate), " loss = ", loss)

        torch.save(net, "%s/demo_network_%d_%d.pkl" % (model_save_dir, epoch,batch_size))
        torch.save(net, "demo_network_final_rnn.pkl")
        batch_size = batch_size + 10
        print("Batch size = ", batch_size)
        # plt.show()
