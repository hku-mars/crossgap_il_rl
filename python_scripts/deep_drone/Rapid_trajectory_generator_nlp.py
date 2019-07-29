import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data as Data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np
import cv2 as cv2
import sys, os
import json

sys.path.append("../query_data/")
from Rapid_trajectory_generator import Rapid_trajectory_generator

class Self_def_loss(torch.nn.Module):
    def __init__(self, weight_vec):
        super(Self_def_loss, self).__init__()
        self.weight_vec = weight_vec
        return

    def forward(self, prediction, target):
        # print("Run forward")
        pos_err_vec = prediction[:,(0,1,2)] - target[:,(0,1,2)]
        spd_err_vec = prediction[:,(3,4,5)] - target[:,(3,4,5)]
        acc_err_vec = prediction[:,(6,7,8)] - target[:,(6,7,8)]
        # print(prediction.shape, " --- ", pos_err_vec.shape)
        # print(pos_err_vec)
        err_vec_withweight_norm = pos_err_vec.norm(p=2, dim=1, keepdim=True)*self.weight_vec[0] + \
                           spd_err_vec.norm(p=2, dim=1, keepdim=True) * self.weight_vec[1] + \
                           acc_err_vec.norm(p=2, dim=1, keepdim=True) * self.weight_vec[2]
        # print("pos_err_vec_norm = ", err_vec_withweight_norm)
        return err_vec_withweight_norm.mean()


if __name__ == "__main__":
    print("hello")
    # model_save_dir = "./rapid_traj_sgd_100_15/"
    json_config = json.load(open("./config/config_rapid_trajectory.json",'r'))
    # model_save_dir = "./rapid_traj_sgd_100_8/"
    model_save_dir = json_config["model_save_dir"]
    print("model_save_dir = ", model_save_dir)
    try:
        os.mkdir(model_save_dir)
    except Exception as e:
        print(e)
    rapid_trajectory = Rapid_trajectory_generator()
    rapid_trajectory_validate = Rapid_trajectory_generator()
    rapid_trajectory_test = Rapid_trajectory_generator()
    sample_size = 20000
    t_start = cv2.getTickCount()
    input_data, output_data = rapid_trajectory.load_from_file("%s/batch_%d.pkl" % (json_config["data_load_dir"],sample_size))

    validation_input_data, validation_output_data = rapid_trajectory_validate.load_from_file("%s/traj_0.pkl"% (json_config["data_load_dir"]))
    plt.ion()
    # rapid_trajectory_validate.plot_data()

    torch_validation_input_data = torch.from_numpy(validation_input_data).float()
    torch_validation_output_data = torch.from_numpy(validation_output_data).float()

    print("Load data cost time  = %.2f " % ((cv2.getTickCount() - t_start) * 1000.0 / cv2.getTickFrequency()))

    # rapid_trajectory.plot_data()
    # plt.pause(0.1)

    n_hidden_per_layer = 100
    net = torch.nn.Sequential(
        torch.nn.Linear(17, n_hidden_per_layer),
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
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, 9)).float().cuda()  # method 2
    bias_net_idx = json_config["bias_net_idx"]
    net = torch.load( "%s/rapid_traj_network_%d.pkl" % (model_save_dir, bias_net_idx))
    print(net)

    # input_data_for_test = np.linspace(-1, 1, 1000)[:,np.newaxis]
    # print(input_data_for_test)
    # output_data_for_test = np.power(input_data_for_test, 2, )

    # torch_input_data = torch.from_numpy(input_data_for_test[:,0][:,np.newaxis]).float().cuda()
    # torch_output_data = torch.from_numpy(output_data_for_test[:,0][:,np.newaxis]).float().cuda()
    torch_input_data = torch.from_numpy(input_data).float()
    torch_output_data = torch.from_numpy(output_data).float()

    # plt.figure("New")
    # plt.plot(torch_input_data.data.cpu().numpy(), torch_output_data.data.cpu().numpy())
    # plt.pause(1)

    print(torch_input_data.shape)
    print(torch_output_data.shape)

    BATCH_SIZE = int(10000 * 10)

    torch_dataset = Data.TensorDataset(torch_input_data.cpu(), torch_output_data.cpu())
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,           # if shuffle the data
        num_workers=7,          # using how many worker.
         )
    print("Loader_size = ", len(loader))

    # learn_rate = 0.0005
    # learn_rate = 0.0001
    learn_rate = 0.000001 * 0.5
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)
    # optimizer = torch.optim.ASGD(net.parameters(), lr=learn_rate)
    # optimizer = torch.optim.SparseAdam(net.parameters(), lr=learn_rate)
    # optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)

    # loss_func = torch.nn.MSELoss().cuda()
    loss_func = Self_def_loss([4,2,1]) # pos is more import with higer weight

    t_start = cv2.getTickCount()
    file = open("%s/train_log.txt"%(model_save_dir),"a")
    save_time = 1000
    for epoch in range(10000):
        print("Epoch = ", epoch)
        learn_rate = learn_rate * 0.998
        print("learn_rate=" ,learn_rate)
        for g in optimizer.param_groups:  # change learning rate
            g['lr'] = learn_rate
        for loop_1, (batch_x, batch_y) in enumerate(loader):
            train_input_data = batch_x.cuda()
            train_target_data = batch_y.cuda()
            loop_2_times = 10000
            for loop_2 in range(loop_2_times):
                prediction = net(train_input_data)
                loss = loss_func(prediction, train_target_data).cuda()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t = epoch * len(loader) + loop_1 * loop_2_times + loop_2 + bias_net_idx
                if (t % 2000 == 0):

                    file.writelines("Learn_rate = " + str(learn_rate) + "\n")

                if (t % (save_time*100) == 0):
                    prediction = net(torch_validation_input_data.cuda())
                    rapid_trajectory_test.input_data = validation_input_data
                    rapid_trajectory_test.output_data = prediction.data.cpu().numpy()
                    fig = plt.figure("test")
                    plt.clf()
                    fig.set_size_inches(16*2, 9*2)
                    rapid_trajectory_validate.plot_data("test")
                    rapid_trajectory_test.plot_data_dash("test")
                    # plt.pause(0.01)
                    fig.savefig("%s/test_%d.png" % (model_save_dir, t / save_time))

                if (t % 10000 == 0):
                    torch.save(net, "rapid_traj_network_final.pkl")
                    print('t =', t, " loss", loss)
                if t % 100 == 0:
                    # print('t =', t, " loss = ", loss.data.cpu().numpy())
                    time = float(int((cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 100) / 100)
                    log_str = str(time) + " |epoch = " + str(epoch) + ' |loop_1 = ' + str(loop_1) + ' |loop_2 = ' + str(loop_2) + ' |t =' + str(t) + " |loss = " + str(loss.data.cpu().numpy())
                    file.writelines(log_str + '\n')
                    # file.pr(log_str+"\r\n")
                    file.flush()
                    print(log_str)
                if (t % 100 == 0) and (t != bias_net_idx):
                    torch.save(net, "%s/rapid_traj_network_%d.pkl" % (model_save_dir, t))

            loop_2_times = loop_2_times - 10
            if (loop_2_times < 1):
                loop_2_times = 1
    torch.save(net, "demo_network_final.pkl")
