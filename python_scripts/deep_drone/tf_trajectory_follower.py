# from tools.img_tools import img_tools
import tensorflow as tf
from tensorflow.contrib import graph_editor as ge
import torch as torch
import torch.utils.data as Data
import numpy as np
import pickle as pkl
import cv2
import sys, os
import json
import copy
import matplotlib.pyplot as plt

work_dir = "../"
# sys.path.append("%s"%work_dir)
sys.path.append("%s/query_data" % work_dir)
sys.path.append("%s/deep_drone" % work_dir)

from Rapid_trajectory_generator import Rapid_trajectory_generator

from trajectory_rnn_net import Rnn_net
from trajectory_data_loder import Trajectory_data_loader
# import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Disable GPU using
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

deep_drone_path = ("%s/deep_drone/" % work_dir)
trajectory_network_name = "%s/rapid_traj_network_8555200.pkl" % deep_drone_path
logger_name = "tf_rnn_log.txt"

logger = open(logger_name,"w")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Trajectory_follower_tf:
    def __init__(self):
        self.if_mode_rnn = 0
        self.input_size = 9
        self.output_size = 4
        self.tf_lay = []
        self.tf_w = []
        self.tf_b = []
        self.train = None
        self.net_input = None
        self.net_output = None
        self.target_output = None  # For supervised trainning
        self.weighted_val = [4, 2, 1]
        self.learn_rate = 0.0001 * 0.5
        self.cell_vec = []
        self.tf_rnn_w = []
        self.tf_rnn_b = []

    def build_rnn_net(self):
        with tf.name_scope("Trajectory_follower_rnn_net"):
            with tf.name_scope("RNN_net"):
                # Rnn lay 0
                for lay_idx in range(2):
                    # continue
                    with tf.name_scope("Rnn_lay_" + str(lay_idx)):
                        num_units = 20
                        cell = tf.contrib.rnn.BasicRNNCell(num_units=20, name="rnn_cell_" + str(lay_idx), activation=None)
                        temp_lay = tf.expand_dims(self.tf_lay[-1], axis=0, name = "expand_dim")

                        input_zero = tf.zeros([1, self.tf_lay[-1].shape[1]], dtype=tf.float32)
                        state_init = cell.zero_state(batch_size=1, dtype=tf.float32)
                        output_temp, state_hidden = cell(inputs=input_zero, state=state_init)

                        cell._kernel = tf.get_variable("rnn_w_" + str(lay_idx), initializer=np.zeros([self.tf_lay[-1].shape[1] + num_units, num_units], dtype = np.float32))
                        cell._bias = tf.get_variable("rnn_b_" + str(lay_idx), initializer=np.zeros([num_units], dtype=np.float32))

                        # rnn_hidden_state_init = tf.placeholder(tf.float32, [1, num_units])
                        rnn_hidden_state_init = tf.placeholder_with_default(np.zeros([1, num_units], dtype=np.float32), [1, num_units])
                        self.rnn_hidden_state_in.append(rnn_hidden_state_init)
                        # print("temp_lay = ", temp_lay)
                        output, state_hidden = tf.nn.dynamic_rnn(cell, temp_lay, initial_state=rnn_hidden_state_init,swap_memory=False)
                        output = tf.squeeze(output, 0)
                        self.rnn_hidden_state_out.append(state_hidden)
                        self.tf_lay.append(output)

            with tf.name_scope("Lay_%s" % len(self.tf_w)):
                self.tf_w.append(tf.Variable(tf.zeros( [20, 4], name="layer_%s_w" % 0)))
                self.tf_b.append(tf.Variable(tf.zeros( [4], name="layer_%s_b" % 0)))
                self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])
        self.net_output = tf.identity(self.tf_lay[-1], name = "Trajectory_follower_rnn_net_out" )
        # self.train_loop()
        # sess.run(tf.global_variables_initializer())
        return self.tf_lay

    def build_mlp_net(self):

        with tf.name_scope("Trajectory_follower_mlp_net"):
            with tf.name_scope("Lay_%s" % len(self.tf_w)):
                module = len(self.tf_w)
                self.tf_w.append(tf.Variable(tf.zeros([9, 20]), name="layer_%d_w" % (int(module))))
                self.tf_b.append(tf.Variable(tf.zeros([20]), name="layer_%d_b" % (int(module))))
                # current_wb_idx = len(tf_lay_b) - 1
                self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])

                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_w[-1])
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_b[-1])
                # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.get_variable(name="layer_%d_b" % (int(module)), regularizer=self.regularizer))

            for loop in range(6):
                with tf.name_scope("Lay_%s" % len(self.tf_w)):
                    module = 2*len(self.tf_w)-1
                    self.tf_w.append(tf.Variable(tf.zeros([20, 20]), name="layer_%d_w" % (int(module))))
                    self.tf_b.append(tf.Variable(tf.zeros([20]), name="layer_%d_b" % (int(module))))
                    # current_wb_idx = len(tf_lay_b) - 1
                    self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])
                    self.tf_lay.append(tf.nn.relu(self.tf_lay[-1], name="relu_%d" % len(self.tf_lay)))

                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_w[-1])
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_b[-1])
                    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.get_variable(self.tf_w[-1], regularizer=self.regularizer))
                    # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.get_variable(self.tf_b[-1], regularizer=self.regularizer))
                # print(loop)
            with tf.name_scope("Lay_%s" % len(self.tf_w)):
                module = 2 * len(self.tf_w) - 1
                self.tf_w.append(tf.Variable(tf.zeros([20, 4]), name="layer_%s_w" % (int(module))))
                self.tf_b.append(tf.Variable(tf.zeros([4]), name="layer_%s_b" % (int(module))))

                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_w[-1])
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_b[-1])

                self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])
        self.net_output = tf.identity(self.tf_lay[-1], name = "Trajectory_follower_mlp_net_out" )
        return self.tf_lay

    def build_net(self, lay_input = None):
        self.input_size = 9
        self.output_size = 4
        self.target_output = tf.placeholder(tf.float32, [None, self.output_size], name='target_output');
        self.tf_lay = []
        self.tf_w = []
        self.tf_b = []
        self.rnn_hidden_state_in = []
        self.rnn_hidden_state_out = []
        self.regularizer = tf.contrib.layers.l1_regularizer(0.1)
        if lay_input is None:
            self.net_input = tf.placeholder(tf.float32, [None, self.input_size], name='net_input');
        else:
            self.net_input = tf.placeholder_with_default(lay_input, [None, self.input_size], name='net_input')
            # self.net_input = lay_input
        self.tf_lay.append(self.net_input)
        if(self.if_mode_rnn):
            print("Build rnn control network")
            return  self.build_rnn_net()
        else:
            print("Build mlp control network")
            return  self.build_mlp_net()

    def torch_net_to_tf_net_mlp(self, net):
        logger.writelines("run torch_net_to_tf_net_mlp")
        print("run torch_net_to_tf_net")
        # self.input_size = net._modules['0'].in_features
        # self.output_size = net._modules[str(len(net._modules) - 1)].out_features
        print("net in size  =  ", self.input_size)  # in size = 17
        print("net out size =  ", self.output_size)  # out size = 9
        self.target_output = tf.placeholder(tf.float32, [None, self.output_size], name='net_input');
        self.tf_lay = []
        self.tf_w = []
        self.tf_b = []
        self.rnn_hidden_state_in = []
        self.rnn_hidden_state_out = []
        # self.net_input = tf.placeholder(tf.float32, [None, self.input_size], name='net_input');
        self.tf_lay.append(self.net_input)
        with tf.name_scope("Trajectory_follower_mlp_net"):
            self.net_input = tf.placeholder(tf.float32, [None, self.input_size], name='net_output');

            self.tf_lay.append(self.net_input)
            for module in net._modules:

                pytoch_lay = net._modules[module]
                if ("Linear" in str(pytoch_lay)):
                    with tf.name_scope("Lay_%s" % len(self.tf_w)):
                        w = pytoch_lay.weight
                        b = pytoch_lay.bias
                        print("layer_%d_w" % (int(module)), "shape = ", w.shape)
                        print("layer_%d_b" % (int(module)), "shape = ", b.shape)
                        self.tf_w.append(tf.Variable(tf.convert_to_tensor(copy.deepcopy(w.cpu().data.numpy().T)), name="layer_%d_w" % (int(module))))
                        self.tf_b.append(tf.Variable(tf.convert_to_tensor(copy.deepcopy(b.cpu().data.numpy().T)), name="layer_%d_b" % (int(module))))

                         # current_wb_idx = len(tf_lay_b) - 1
                        self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])

                        print(self.tf_lay[-1])
                        print(self.tf_w[-1])
                        print(self.tf_b[-1])
                        # print("[Linear]: w shape = ", w.shape, " b shape = ", b.shape)
                        # break
                else:
                    self.tf_lay.append(tf.nn.relu(self.tf_lay[-1], name="relu_%d" % len(self.tf_lay)))
                    # last_tf_lay = tf.nn.relu(last_tf_lay)
                    # print("[ Relu ]")
                # print(str(net._modules[module]))
        self.net_output = self.tf_lay[-1]
        # self.train_loop()
        # sess.run(tf.global_variables_initializer())
        return self.tf_lay

    def torch_net_to_tf_net_rnn(self, net):
        logger.writelines("run torch_net_to_tf_net")
        print("run torch_net_to_tf_net")
        # self.input_size = net._modules['0'].in_features
        # self.output_size = net._modules[str(len(net._modules) - 1)].out_features
        print("net in size  =  ", self.input_size)  # in size = 17
        print("net out size =  ", self.output_size)  # out size = 9
        self.target_output = tf.placeholder(tf.float32, [None, self.output_size], name='net_input');
        self.tf_lay = []
        self.tf_w = []
        self.tf_b = []
        self.rnn_hidden_state_in = []
        self.rnn_hidden_state_out = []
        self.net_input = tf.placeholder(tf.float32, [None, self.input_size], name='net_input');
        self.tf_lay.append(self.net_input)

        with tf.name_scope("Trajectory_follower_rnn_net"):
            for module in net._modules:
                pytoch_lay = net._modules[module]
                if ("Linear" in str(pytoch_lay)):
                    with tf.name_scope("Lay_%s" % len(self.tf_w)):
                        w = pytoch_lay.weight
                        b = pytoch_lay.bias
                        self.tf_w.append(tf.Variable(tf.convert_to_tensor(copy.deepcopy(w.cpu().data.numpy().T), name="layer_%s_w" % (str(module)))))
                        self.tf_b.append(tf.Variable(tf.convert_to_tensor(copy.deepcopy(b.cpu().data.numpy().T), name="layer_%s_b" % (str(module)))))
                        self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])

                elif("RNN" in str(pytoch_lay)):
                    numer_layer = pytoch_lay.num_layers
                    logger.writelines("Net have RNN module, layers number = "+ str(numer_layer))
                    print("Net have RNN module, layers number = ", numer_layer)
                    with tf.name_scope("RNN_net" ):
                        for lay_idx in range(numer_layer):
                            with tf.name_scope("Rnn_lay_"+str(lay_idx)):
                                w_ih_name = "weight_ih_l"+str(lay_idx)
                                w_hh_name = "weight_hh_l"+str(lay_idx)
                                b_ih_name = "bias_ih_l"+str(lay_idx)
                                b_hh_name = "bias_hh_l"+str(lay_idx)

                                print("=== Lay%d, "%lay_idx, w_ih_name, w_hh_name, b_ih_name, b_hh_name," ===" )
                                w_ih = pytoch_lay._parameters[w_ih_name].data.cpu().data.numpy()
                                w_hh = pytoch_lay._parameters[w_hh_name].data.cpu().data.numpy()
                                b_ih = pytoch_lay._parameters[b_ih_name].data.cpu().data.numpy()
                                b_hh = pytoch_lay._parameters[b_hh_name].data.cpu().data.numpy()
                                tf_np_weight = np.concatenate((w_ih, w_hh), 1).T
                                tf_np_bias   = b_ih + b_hh
                                num_units = w_ih.shape[0]

                                cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units, name="rnn_cell_" + str(lay_idx), activation = None)

                                temp_lay = tf.expand_dims(self.tf_lay[-1], axis=0)

                                input_zero = tf.zeros([1, self.tf_lay[-1].shape[1]], dtype=tf.float32)
                                state_init = cell.zero_state(batch_size=1, dtype=tf.float32)
                                output_temp, state_hidden = cell(inputs=input_zero, state=state_init )

                                cell._kernel = tf.get_variable("rnn_w_" + str(lay_idx), initializer=copy.deepcopy(tf_np_weight))
                                cell._bias = tf.get_variable("rnn_b_" + str(lay_idx), initializer=copy.deepcopy(tf_np_bias.T))

                                rnn_hidden_state_init = tf.placeholder(tf.float32, [1,num_units] )
                                self.rnn_hidden_state_in.append(rnn_hidden_state_init)
                                output, state_hidden = tf.nn.dynamic_rnn(cell, temp_lay, initial_state=rnn_hidden_state_init)
                                output = tf.squeeze(output, 0)
                                self.rnn_hidden_state_out.append(state_hidden)
                                self.tf_lay.append(output)
                                continue
                else:
                    self.tf_lay.append(tf.nn.relu(self.tf_lay[-1], name="relu_%d" % len(self.tf_lay)))

        self.net_output = self.tf_lay[-1]
        # self.train_loop()
        # sess.run(tf.global_variables_initializer())
        return self.tf_lay

    def train_loop(self):
        print(self.net_output, self.target_output)
        diff = self.net_output - self.target_output

        rp_rr = tf.slice(diff, [0, 0], [-1, 3])
        yaw_rr = tf.slice(diff, [0, 2], [-1, 1])
        thr_err = tf.slice(diff, [0, 3], [-1, 1])

        # for weight in self.tf_w:
        #     weight_temp = tf.get_variable(weight, regularizer=self.regularizer)
        #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_temp)

        reg_variables =  tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.train_loss = tf.reduce_mean(tf.abs(rp_rr)+tf.abs(yaw_rr)*0.1+tf.abs(thr_err )*5 , name="loss")
        print(self.regularizer)
        print(reg_variables)

        self.total_loss = self.train_loss + tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)*0.0001
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)

if __name__ == "__main__":
    print("this is tf_trajectory_follower")
    trajectory_data_test_set = Trajectory_data_loader()
    trajectory_data_train_set = Trajectory_data_loader()
    torch.manual_seed(1)  # reproducible
    if_mode_rnn = 0
    import json
    json_config = json.load(open("./tf_train_config/tf_trajectory_follower.json",'r'))
    print(json_config)
    test_logger_name_vec = json_config['test_log']
    train_logger_name_vec =  json_config['train_log']
    print("test__set = " , test_logger_name_vec)
    print("train_set = " , train_logger_name_vec)

    model_save_dir = './model_temp_mlp_batch'
    try:
        os.mkdir(model_save_dir)
    except Exception as e:
        print(e)

    # trajectory_data_loader.plot_data(0)
    # plt.show()
    # rnn_net_for_eval = Rnn_net()
    if(if_mode_rnn):
        rnn_net_for_eval = torch.load("demo_network_final_rnn.pkl")
        print(rnn_net_for_eval)
        print("-----------------")
        # trajectory_data_loader.eval_net_rnn(rnn_net_for_eval, 0 )
        # trajectory_data_loader.plot_data(0)
        np.set_printoptions(precision=3)
        traj_follower =  Trajectory_follower_tf()
        traj_follower.build_net();
    else:
        # torch_net = torch.load("../deep_drone/demo_network_50000.pkl")
        # print(torch_net)
        traj_follower = Trajectory_follower_tf()
        # traj_follower.torch_net_to_tf_net_mlp(torch_net)
        traj_follower.build_net();
        print("-----------------")

    traj_follower.learn_rate = json_config['lr']


    # traj_follower.torch_net_to_tf_net(rnn_net_for_eval)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    if(if_mode_rnn):
        tf.summary.FileWriter("rnn_tf_log/", sess.graph)
    else:
        tf.summary.FileWriter("mlp_tf_log_temp/", sess.graph)

    if (if_mode_rnn):
        net_save_dir = "./tf_rnn_net"
    else:
        net_save_dir = "./tf_mlp_net"

    try:
        os.makedirs( net_save_dir )
    except:
        pass
    saver = tf.train.Saver(max_to_keep=50000)
    if(if_mode_rnn):
        # save_path = saver.save(sess, "%s/save_net_rnn.ckpt"%net_save_dir)
        saver.restore(sess,"%s/save_net_rnn.ckpt"%net_save_dir)
    else:
        # save_path = saver.save(sess, "%s/save_net_mlp.ckpt"%net_save_dir)
        saver.restore(sess,"%s/save_net_mlp.ckpt"%net_save_dir)

    print("Restore parameters finish")
    # print(save_path)

    # exit( 0 )

    logger.flush()


    input_data_for_test, output_data_for_test = trajectory_data_test_set.load_data_from_logger_vec(test_logger_name_vec)

    # logger.writelines(str(cell.variables[1].eval()) + "\n")

    # tf_output_data = sess.run(traj_follower.net_output, feed_dict={traj_follower.net_input: input_data_for_test*1})
    tf_output_data = output_data_for_test * 1

    # the following method are the same.
    if(if_mode_rnn):
        feed_hidden_state_0 = np.zeros(traj_follower.rnn_hidden_state_in[0].shape)
        feed_hidden_state_1 = np.zeros(traj_follower.rnn_hidden_state_in[1].shape)
        # tf_output_data = sess.run(traj_follower.net_output , feed_dict={traj_follower.net_input: np.matrix(input_data_for_test),
        #                                                                     traj_follower.rnn_hidden_state_in[0]: feed_hidden_state_0,
        #                                                                     traj_follower.rnn_hidden_state_in[1]: feed_hidden_state_1})

        tf_output_data = sess.run(traj_follower.net_output, feed_dict={traj_follower.net_input: np.matrix(input_data_for_test)})

        # for k in range(input_data_for_test.shape[0]):
        #     # tf_output_data[k,:] = sess.run(traj_follower.net_output, feed_dict={traj_follower.net_input: np.matrix(input_data_for_test[k,:])}).ravel()
        #     [tf_output_data[k,:],feed_hidden_state_0,feed_hidden_state_1] = sess.run([traj_follower.net_output,
        #                                     traj_follower.rnn_hidden_state_out[0],
        #                                     traj_follower.rnn_hidden_state_out[1]] , feed_dict={traj_follower.net_input: np.matrix(input_data_for_test[k,:]),
        #                                                                                          traj_follower.rnn_hidden_state_in[0]: feed_hidden_state_0,
        #                                                                                          traj_follower.rnn_hidden_state_in[1]: feed_hidden_state_1})
    else:
        tf_output_data = sess.run(traj_follower.net_output, feed_dict={traj_follower.net_input: np.matrix(input_data_for_test)})

    print("===== test_set =====")
    print(tf_output_data.shape)
    print("raw_mean = ", np.mean(np.abs(output_data_for_test)))
    print("err_mean = ", np.mean(np.abs(tf_output_data - output_data_for_test)))
    # print("rnn_2 output = ", np.mean(np.abs(traj_follower.rnn_output[1].eval())))
    # exit()
    # trajectory_data_test_set.plot_data(0)
    #
    trajectory_data_test_set.eval_net(tf_output_data, 0)
    plt.show()

    # =================== Learn =========================#

    print("Load data form ", test_logger_name_vec)
    input_data_for_train, output_data_for_train = trajectory_data_train_set.load_data_from_logger_vec(train_logger_name_vec)
    print("train_set_size = ", input_data_for_train.shape)
    traj_follower.train_loop()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    bias_net_idx = json_config['bias_idx']
    if(bias_net_idx!=0):
        saver.restore(sess, "./tf_follower_mlp/tf_saver_%d.ckpt" % bias_net_idx)
    else:
        saver.restore(sess,"%s/save_net_mlp.ckpt"%net_save_dir)

    tf.summary.FileWriter("mlp_tf_log_temp/", sess.graph)

    torch_input_data = torch.from_numpy(input_data_for_train).float()
    torch_output_data = torch.from_numpy(output_data_for_train).float()

    # BATCH_SIZE = int(1500 * 1.3 * 100)
    BATCH_SIZE = json_config["batch_size"]
    print("Total data size = ", input_data_for_train.shape)
    print("Batch size = ", BATCH_SIZE)
    torch_dataset = Data.TensorDataset(torch_input_data.cpu(), torch_output_data.cpu())
    loader = Data.DataLoader(
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=BATCH_SIZE,  # mini batch size
            shuffle=True,  # if shuffle the data
            num_workers=json_config["shuffle_thread"],  # using how many worker.
    )

    print("Load finish.")
    print("Set data finish.")
    print("Batch number = ", len(loader))
    save_idx = 100

    for epoch in range(10000):
        for loop_1, (batch_x, batch_y) in enumerate(loader):
            np_data_x = batch_x.data.numpy()
            np_data_y = batch_y.data.numpy()
            loop_2_times = 100000
            for loop_2 in range(loop_2_times):
                traj_follower.train_step.run({traj_follower.net_input: np.matrix(np_data_x), traj_follower.target_output: np.matrix(np_data_y)})
                t = epoch * len(loader)*loop_1*loop_2_times + loop_1 * loop_2_times + loop_2 + bias_net_idx
                if ( (t % save_idx == 0) and t != bias_net_idx):
                    try:
                        save_path = saver.save(sess, "./tf_follower_mlp/tf_saver_%d.ckpt" % t)
                        print(save_path)
                    except Exception as e:
                        print("Save net errror")
                        print(e)

                if (t % (10) == 0):

                    loss = sess.run(traj_follower.train_loss, feed_dict={traj_follower.net_input: np.matrix(np_data_x),
                                                                          traj_follower.target_output: np.matrix(np_data_y)})
                    loss_total = sess.run(traj_follower.total_loss, feed_dict={traj_follower.net_input: np.matrix(np_data_x),
                                                                          traj_follower.target_output: np.matrix(np_data_y)})
                    loss_test = sess.run(traj_follower.train_loss, feed_dict={traj_follower.net_input: np.matrix(input_data_for_test),
                                                                               traj_follower.target_output: np.matrix(output_data_for_test)})
                    # error_test = traj_follower.train_loss.run({traj_follower.net_input: np.matrix(input_data_for_test), traj_follower.target_output: np.matrix(output_data_for_test)})

                    error = np.mean(np.abs(sess.run(traj_follower.net_output, feed_dict={traj_follower.net_input: np_data_x}) - np_data_y))
                    error_test = np.mean(np.abs(sess.run(traj_follower.net_output, feed_dict={traj_follower.net_input: np.matrix(input_data_for_test)}) - output_data_for_test))
                    log_str = "epoch = " + str(epoch) + ' |loop_1 = ' + str(loop_1) + ' |loop_2 = ' + str(loop_2) + ' |t =' + str(t) \
                              + " |loss = " + "%.3f"%loss + " |loss_total = " + "%.3f"%loss_total \
                              + " |error= " + "%.3f"%error + " |errot_test = " + "%.3f"%error_test \

                    print(log_str)
                # if (t % save_idx == 0):
                #     rapid_trajectory_validate.output_data = validation_output_data
                #     rapid_trajectory_validate.plot_data("test")
                #     tf_out_data = sess.run(tf_net.net_output, feed_dict={tf_net.net_input: validation_input_data})
                #     rapid_trajectory_validate.output_data = tf_out_data
                #     rapid_trajectory_validate.plot_data_dash("test")
                #     fig.savefig("%s/test_%d.png" % (save_dir, t))

    print("Finish")
    exit()
