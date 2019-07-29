import tensorflow as tf
import torch as torch
import torch.utils.data as Data
import numpy as np
import pickle as pkl
import cv2
import sys, os
import json
import copy
import math
import matplotlib.pyplot as plt
import re
import random
import torch
work_dir = "../"
# sys.path.append("%s"%work_dir)
sys.path.append("%s/query_data" % work_dir)

from Rapid_trajectory_generator import Rapid_trajectory_generator

deep_drone_path = ("%s/deep_drone/" % work_dir)
trajectory_network_name = "%s/rapid_traj_network_8555200.pkl" % deep_drone_path

# Input[17]: time_1, cross_spd_1, pos_error_3, spd_start_3, acc_start_3, spd_end_3, acc_end_3
#                 0,           1, 2         4, 5         7, 8        10, 11     13, 14     16
# Output[9]: tar_pos_3, tar_spd_3, tar_acc_3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Disable GPU using
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

IF_DATA_ARGU = 1
IF_NORM = 1

def resort_para_form_checkpoint( prefix, graph, sess, ckpt_name ):
    from tensorflow.python import pywrap_tensorflow
    # ckpt_name_vec = ["./tf_net/planning_net/tf_saver_252750.ckpt", "./tf_net/control_mlp_net_train/tf_saver_2318100.ckpt"]
    # ckpt_name_vec = ["./tf_net/planning_net/tf_saver_252750.ckpt", "./tf_net/control_mlp_net_train/tf_saver_1300000.ckpt"]
    # ckpt_name_vec = ["./tf_net/planning_net/tf_saver_252750.ckpt", "./tf_net/control_mlp_net/save_net_mlp.ckpt"]
    ckpt_name_vec = [ckpt_name]
    print("=========")
    file = open("full_structure.txt","w")
    file.writelines(str(graph.get_operations()))
    # for ops in tf.Graph.get_all_collection_keys():
    # for ops in graph.get_operations():
    #     file.writelines(ops)
    #     print(ops)
    file.close()
    print("=========")
    with tf.name_scope("restore"):
        for ckpt_name in ckpt_name_vec:
            print("===== Restore data from %s =====" % ckpt_name)
            reader = pywrap_tensorflow.NewCheckpointReader(ckpt_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            for _key in var_to_shape_map:
                # print("tensor_name: ", key)
                # print(reader.get_tensor(key))
                # tensor = graph.get_tensor_by_name(key)
                key = prefix + _key+ ":0"
                # key = prefix + _key
                try:
                    tensor = graph.get_tensor_by_name(key)
                    sess.run(tf.assign(tensor, reader.get_tensor(_key)))
                    # print(tensor)
                except Exception as e:
                    print(key, " can not be restored, e= ",str(e))
                    pass


class Plannning_net_tf:
    def __init__(self):
        self.if_normalize = IF_NORM
        self.tf_lay = []
        self.tf_w = []
        self.tf_b = []
        self.train = None
        self.net_input = None
        self.net_output = None
        self.target_output = None  # For supervised trainning
        self.weighted_val = [2, 1, 1]
        self.learn_rate = 5.0e-6
        self.if_random_init = 0
        # self.learn_rate = 5.0e-6
        self.rapid_generator = Rapid_trajectory_generator() # For conviennient to use

    def train_loop(self):
        print(self.net_output, self.target_output)
        # self.target_output_no_g = self.target_output - tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 9.8])
        # if(self.if_normalize):
        #     diff = tf.divide(self.net_output - self.target_output, self.scale_factor)
        #     # diff = self.net_output - self.target_output_no_g
        # else:
        #     diff =  self.net_output - self.target_output_no_g

        diff = self.net_output - self.target_output
        # if(self.if_normalize):
        # if(self.if_normalize):
        if(0):
        # if(self.if_normalize):
            pos_err = tf.divide( tf.slice(diff, [0, 0], [-1, 3]), self.scale_factor )
            spd_err = tf.slice(diff, [0, 3], [-1, 3])
            acc_err = tf.slice(diff, [0, 6], [-1, 3])
            # spd_err = tf.divide( tf.slice(diff, [0, 3], [-1, 3]), self.scale_factor_spd )
            # acc_err = tf.divide( tf.slice(diff, [0, 6], [-1, 3]), self.scale_factor_acc )
        else:
            pos_err = tf.slice(diff, [0, 0], [-1, 3])
            spd_err = tf.slice(diff, [0, 3], [-1, 3])
            acc_err = tf.slice(diff, [0, 6], [-1, 3])
        try:
            # Capable with older tensorflow keepdims=> keep_dims (https://github.com/openai/pixel-cnn/issues/29)
            self.train_loss = tf.reduce_mean(tf.norm(pos_err, axis=1, keepdims=True) * self.weighted_val[0] + \
                  tf.norm(spd_err, axis=1, keepdims=True) * self.weighted_val[1] + \
                  tf.norm(acc_err, axis=1, keepdims=True) * self.weighted_val[2],
                  name="loss")
        except:
            self.train_loss = tf.reduce_mean(tf.norm(pos_err, axis=1, keep_dims=True) * self.weighted_val[0] + \
                  tf.norm(spd_err, axis=1, keep_dims=True) * self.weighted_val[1] + \
                  tf.norm(acc_err, axis=1, keep_dims=True) * self.weighted_val[2],
                  name="loss")

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = self.train_loss + tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)

        # betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0, amsgrad = False
        # self.train_step = tf.train.AdamOptimizer(self.learnning_rate, epsilon=1e-8, beta1=0.9, beta2=0.999).minimize(self.loss)
        self.train_step = tf.train.AdamOptimizer(self.learnning_rate).minimize(self.loss)
        # self.train_step = tf.train.GradientDescentOptimizer(self.learnning_rate).minimize(self.loss)
        # self.train_step = tf.train.MomentumOptimizer(self.learnning_rate/1000,self.learnning_rate/10.0).minimize(self.loss)
        # self.train_step = tf.train.RMSPropOptimizer(self.learnning_rate,self.learnning_rate/10.0).minimize(self.loss)
        # self.train_step = tf.train.AdagradOptimizer(self.learnning_rate).minimize(self.loss)

    def build_net(self, X = None):
        self.input_size = 17
        self.output_size = 9
        self.learnning_rate = tf.placeholder(tf.float32, shape=[])
        self.target_output = tf.placeholder(tf.float32, [None, self.output_size], name='target_output');

        # self.regularizer = tf.contrib.layers.l1_regularizer(1.0e-10)
        self.regularizer = tf.contrib.layers.l1_regularizer(0.0e-10)

        self.tf_lay = []
        self.tf_w = []
        self.tf_b = []


        with tf.name_scope("net"):
            if X is None:
                self.net_input = tf.placeholder(tf.float32, [None, self.input_size], name='net_input');
            else:
                self.net_input = tf.placeholder_with_default(X, [None, self.input_size], name='net_input');
            self.tf_lay.append(self.net_input)

            if (self.if_normalize):
                with tf.name_scope("normalize"):
                    try:
                        # Capable with older tensorflow keepdims=> keep_dims (https://github.com/openai/pixel-cnn/issues/29)
                        self.pos_scale_nor = tf.norm(tf.slice(self.net_input, [0, 2], [-1, 3]), axis=1, keepdims=True, name="scale_factor")
                        self.scale_factor = tf.divide( tf.norm(tf.slice(self.net_input, [0, 2], [-1, 3]), axis=1, keepdims=True, name="scale_factor") ,
                                                       tf.slice(self.net_input, [0, 1], [-1, 1]) )*1
                    except: 
                        self.pos_scale_nor = tf.norm(tf.slice(self.net_input, [0, 2], [-1, 3]), axis=1, keep_dims=True, name="scale_factor")
                        self.scale_factor = tf.divide( tf.norm(tf.slice(self.net_input, [0, 2], [-1, 3]), axis=1, keep_dims=True, name="scale_factor") ,
                                                       tf.slice(self.net_input, [0, 1], [-1, 1]) )*1

                    self.scale_factor_spd = tf.multiply(self.scale_factor, self.scale_factor)
                    self.scale_factor_acc = tf.multiply(self.scale_factor_spd, self.scale_factor)

                    no_scaled_tensor = tf.slice(self.tf_lay[-1], [0, 0], [-1, 2])

                    with tf.name_scope("pos_divide_scale"):
                        scaled_pos = tf.multiply( tf.slice(self.tf_lay[-1], [0, 2], [-1, 3]) ,  self.scale_factor)
                    with tf.name_scope("spd_divide_scale"):
                        scaled_spd_start = tf.multiply( tf.slice(self.tf_lay[-1], [0, 5], [-1, 3]), self.scale_factor_spd )
                        scaled_spd_end = tf.multiply( tf.slice(self.tf_lay[-1], [0, 11], [-1, 3]), self.scale_factor_spd )
                        # scaled_spd_start = tf.divide( tf.slice(self.tf_lay[-1], [0, 5], [-1, 3]), self.scale_factor )
                        # scaled_spd_end = tf.divide( tf.slice(self.tf_lay[-1], [0, 11], [-1, 3]), self.scale_factor )
                    with tf.name_scope("acc_divide_scale"):
                        scaled_acc_start = tf.multiply( tf.slice(self.tf_lay[-1], [0, 8], [-1, 3]), self.scale_factor_acc )
                        scaled_acc_end = tf.multiply( tf.slice(self.tf_lay[-1], [0, 14], [-1, 3]), self.scale_factor_acc )
                        # scaled_acc_start = tf.divide( tf.slice(self.tf_lay[-1], [0, 8], [-1, 3]), self.scale_factor )
                        # scaled_acc_end = tf.divide( tf.slice(self.tf_lay[-1], [0, 14], [-1, 3]), self.scale_factor )

                    # self.tf_lay[-1] =
                    self.input_normalize = tf.concat([no_scaled_tensor,scaled_pos, scaled_spd_start,scaled_acc_start,  scaled_spd_end, scaled_acc_end], 1 , name="input_normalize")
                    self.tf_lay.append(self.input_normalize  )
                    print(no_scaled_tensor)
                    # print(scaled_tensor)
                    # print(self.tf_lay[-1])

            with tf.name_scope("Lay_%s" % len(self.tf_w)):
                if (self.if_random_init):
                    self.tf_w.append(tf.Variable(tf.random_normal([17, 100], dtype=np.float32, name="layer_%d_w" % 0, seed=1)))  # (np.zeros([17, 100], ), name="layer_%d_w" % 0)))
                    self.tf_b.append(tf.Variable(tf.random_normal([100], dtype=np.float32, name="layer_%d_b" % 0, seed=1)))
                else:
                    self.tf_w.append(tf.Variable(tf.zeros([17, 100], dtype=np.float32, name="layer_%d_w" % 0 )))  # (np.zeros([17, 100], ), name="layer_%d_w" % 0)))
                    self.tf_b.append(tf.Variable(tf.zeros([100], dtype=np.float32, name="layer_%d_b" % 0 )))
                self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])

                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_w[-1])
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_b[-1])
                # print(self.tf_lay[-1])

            for idx in range(8):  # 8 X 100
                with tf.name_scope("Lay_%s" % len(self.tf_w)):
                    if(self.if_random_init):
                        self.tf_w.append(tf.Variable(tf.random_normal([100, 100], dtype=np.float32, name="layer_%d_w" % (2 * idx + 0), seed=1)))
                        self.tf_b.append(tf.Variable(tf.random_normal([100], dtype=np.float32, name="layer_%d_b" % (2 * idx + 1), seed=1)))
                    else:
                        self.tf_w.append(tf.Variable(tf.zeros([100, 100], dtype=np.float32, name="layer_%d_w" % (2 * idx + 0) )))
                        self.tf_b.append(tf.Variable(tf.zeros([100], dtype=np.float32, name="layer_%d_b" % (2 * idx + 1) )))
                    self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])

                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_w[-1])
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_b[-1])

                self.tf_lay.append(tf.nn.relu(self.tf_lay[-1], name="relu_%d" % len(self.tf_lay)))

            with tf.name_scope("Lay_%s" % len(self.tf_w)):
                if(self.if_random_init):
                    self.tf_w.append(tf.Variable(tf.random_normal([100, 9], dtype=np.float32, name="layer_%d_w" % (int(17)), seed=1)))
                    self.tf_b.append(tf.Variable(tf.random_normal([9], dtype=np.float32, name="layer_%d_b" % (int(17)), seed=1)))
                else:
                    self.tf_w.append(tf.Variable(tf.zeros([100, 9], dtype=np.float32, name="layer_%d_w" % (int(17)) )))
                    self.tf_b.append(tf.Variable(tf.zeros([9], dtype=np.float32, name="layer_%d_b" % (int(17)) )))
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_w[-1])
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_b[-1])

                self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])

            if (self.if_normalize):
                with tf.name_scope("recover_scale"):
                    self.output_for_train = self.tf_lay[-1]

                    recover_pos = tf.divide(tf.slice(self.tf_lay[-1], [0, 0], [-1, 3]), self.scale_factor, name="divide_scale")
                    recover_spd = tf.divide(tf.slice(self.tf_lay[-1], [0, 3], [-1, 3]), self.scale_factor_spd, name="divide_scale")
                    recover_acc = tf.divide(tf.slice(self.tf_lay[-1], [0, 6], [-1, 3]), self.scale_factor_acc, name="divide_scale")
                    # self.tf_lay[-1] =
                    # self.tf_lay.append( tf.multiply(self.tf_lay[-1], self.scale_factor, name="recover_scale") )
                    temp = tf.concat([recover_pos,recover_spd, recover_acc], axis=1)
                    # temp = temp + tf.multiply( tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 9.8]) ,  1-self.scale_factor )
                    self.tf_lay.append(temp)
                    print(self.tf_lay[-1])

        self.net_output = tf.identity( self.tf_lay[-1], name = "planning_net_out")
        return self.tf_lay

    def pytorch_net_to_tf(self, net):
        self.input_size = net._modules['0'].in_features
        self.output_size = net._modules[str(len(net._modules) - 1)].out_features
        print("net in size  =  ", self.input_size)  # in size = 17
        print("net out size =  ", self.output_size)  # out size = 9
        self.target_output = tf.placeholder(tf.float32, [None, self.output_size], name='net_input');
        self.tf_lay = []
        self.tf_w = []
        self.tf_b = []

        with tf.name_scope("net"):
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
                        self.tf_w.append(tf.Variable(tf.convert_to_tensor(copy.deepcopy(w.cpu().data.numpy().T), name="layer_%d_w" % (int(module)))))
                        self.tf_b.append(tf.Variable(tf.convert_to_tensor(copy.deepcopy(b.cpu().data.numpy().T), name="layer_%d_b" % (int(module)))))
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

    def load_from_pytorch_net(self, trajectory_network_name):
        import torch
        torch_policy_net = torch.load(trajectory_network_name)
        net = torch_policy_net
        self.pytorch_net_to_tf(net)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=50000000)
        save_path = saver.save(sess, "./form_torch_net.ckpt")

def load_example_file_vec(dir):
    file_list = []
    for idx in range(5):
        file_name = "%s/traj_example_%d.pkl"%(dir,idx)
        file_list.append(file_name)
    return  file_list

def traj_augment(input_data, output_data, if_sub_g = 0, traj_size = 1000):

    # Input[17]: time_1, cross_spd_1, pos_error_3, spd_start_3, acc_start_3, spd_end_3, acc_end_3
    #                 0,           1, 2         4, 5         7, 8        10, 11     13, 14     16
    # Output[9]: tar_pos_3, tar_spd_3, tar_acc_3

    # import copy
    t_start = cv2.getTickCount()
    if_regenerate = 0
    if_debug = 0
    if_symetry = 1
    if_reverse = 0
    if_pos_scale = 1
    if_time_scale = 0
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
        scale_para = traj_index = np.random.uniform(low=1, high=5.0, size=1)[0]
        # print("time scale  = ", scale_para)
        scale_factor = [scale_para, 1.0/scale_para]
        for scale in scale_factor:
            enh_input_data = copy.deepcopy(input_data)
            enh_output_data = copy.deepcopy(output_data)

            enh_input_data[:, 1] = enh_input_data[:, 1] * scale
            for traj_idx in range(traj_count):
                enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:, range(0,3)] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:, range(0,3)]
                enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:, range(3,6)] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:, range(3,6)]/scale_para
                enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:, range(6,9)] = enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)][:, range(6,9)]/scale_para/scale_para
                if (if_regenerate):
                    rapid_trajectory.generate_trajectory_np_array( enh_input_data[traj_idx*traj_size + 1] )
                    enh_input_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] = rapid_trajectory.input_data
                    enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] = rapid_trajectory.output_data
                    print("time compare = ", np.mean(np.abs(enh_output_data[range(traj_idx * traj_size, (traj_idx + 1) * traj_size)] - rapid_trajectory.output_data)))
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


if __name__ == "__main__":
    json_file_name = "%s/config/config_rapid_trajectory.json" % deep_drone_path
    print("Load json form :", json_file_name)
    json_config = json.load(open(json_file_name, 'r'))
    print(json_config)
    rapid_trajectory_validate = Rapid_trajectory_generator()
    rapid_trajectory_cross = Rapid_trajectory_generator()

    save_dir = json_config['net_save_dir'] + "_" + str(IF_NORM) + "_" + str(IF_DATA_ARGU)
    log_file_name = json_config['log_file_name'] + "_" + str(IF_NORM) + "_" + str(IF_DATA_ARGU) + ".log"

    try:
        os.makedirs(save_dir)
    except:
        pass
    # torch_policy_net = torch.load(trajectory_network_name)
    # net = torch_policy_net
    # print(torch_policy_net)
    print("----- Print net -----")
    tf_net = Plannning_net_tf()
    # tf_net.load_from_pytorch_net("rapid_traj_network_seed_27.pkl")
    # exit(0)
    tf_lay = tf_net.build_net();

    # tf_lay = tf_net.pytorch_net_to_tf(net)
    # exit(0)
    tf_net.train_loop()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.00
    config.gpu_options.allow_growth = True
    # config.gpu_options.allow_soft_placement = True
    # sess = tf.Session(config=config)
    # sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # sess = tf.InteractiveSession(config=config)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter("./log/", sess.graph)

    saver = tf.train.Saver(max_to_keep=500000000)
    # save_path = saver.save(sess, "./save_net_new.ckpt")
    # print("save_path = ", save_path)
    # exit(0)

    print(deep_drone_path)
    print(trajectory_network_name)

    # validation_input_data, validation_output_data = rapid_trajectory_validate.load_from_file(
    #         "%s/traj_0.pkl" % json_config["data_load_dir"])

    validation_input_data, validation_output_data = rapid_trajectory_validate.load_from_file("%s/traj_0.pkl" % json_config["data_load_dir"])
    # validation_input_data, validation_output_data = rapid_trajectory_validate.load_from_file("%s/traj_cross_no_g.pkl" % json_config["data_load_dir"])
    # validation_input_data, validation_output_data = rapid_trajectory_validate.load_from_file("%s/traj_15015.pkl" % json_config["data_load_dir"])
    # validation_input_data, validation_output_data = rapid_trajectory_validate.load_from_file("%s/traj_cross.pkl" % json_config["data_load_dir"])
    # validation_input_data, validation_output_data = rapid_trajectory_validate.load_from_file("%s/traj_cross.pkl" % json_config["data_load_dir"])
    # validation_input_data, validation_output_data = rapid_trajectory_validate.load_from_file_vector(load_example_file_vec( json_config["data_load_dir"] ), if_limit=0)
    # validation_input_data, validation_output_data = rapid_trajectory_validate.load_from_file_vector(load_example_file_vec( json_config["data_load_dir"] ), if_limit=1)

    # cross_input_data, cross_output_data = rapid_trajectory_cross.load_from_file("%s/traj_cross.pkl" % json_config["data_load_dir"])
    # cross_input_data, cross_output_data = rapid_trajectory_cross.load_from_file("%s/traj_0.pkl" % json_config["data_load_dir"])
    cross_input_data, cross_output_data = rapid_trajectory_cross.load_from_file_vector(load_example_file_vec( json_config["data_load_dir"] ), if_limit=0)

    # bias_net_idx = 252750
    # bias_net_idx = 7000000
    # bias_net_idx = 7570000
    # bias_net_idx = 8262000  # Before change network type
    # bias_net_idx = 9658000  # Before adding example
    # bias_net_idx = 10386000  # Before subtract gravity
    # bias_net_idx = 12135000  # Before argument
    # bias_net_idx = 12844000  # Before argument
    # bias_net_idx = 13478000  # Before add pos_scale
    # bias_net_idx = 13702000  # Before add time_scale
    # bias_net_idx = 13802000  # Before add time_scale
    # bias_net_idx = 15215000  # Before add random scale
    # bias_net_idx = 17100000  # Before add random scale
    # bias_net_idx = 18134000  # Before add random scale
    # bias_net_idx = 21157000  # before fix time_scale error
    # bias_net_idx = 21421000
    # bias_net_idx = 21550000
    # bias_net_idx = 23632000     # Final version
    # bias_net_idx = 27544000     # 1.14!
    # bias_net_idx = 30380000     # 0.7~0.9!!!
    # bias_net_idx = 30440000
    bias_net_idx = 40640000       # 0.64!!!
    bias_net_idx = 44020000       # 0.60!!! before 100 scale
    bias_net_idx = 47290000       # 0.60!!! before 100 scale
    bias_net_idx = 91170000       # 0.95 100 scale
    bias_net_idx = 97020000
    bias_net_idx = 103900000
    bias_net_idx = 143050000     # 0.44
    bias_net_idx = 53248000
    # bias_net_idx = 20000       # 0.64!!!
    if(bias_net_idx!=0):
        print('Append log to ', json_config['log_file_name'])
        log_file = open(log_file_name, 'r')   # Append log to file
        last_line = log_file.readlines()[-1]
        print("Log last line  = ", last_line)
        # print(re.findall(r'\d+.?\d*', last_line))
        log_file.close()
        log_file = open(log_file_name, 'a')  # Append log to file
        log_file.write('\n')
        log_file.flush()
        saver.restore(sess, save_dir + "/tf_saver_" + str(bias_net_idx) + ".ckpt")
        time_bias = float( re.findall(r'\d+.?\d*', last_line)[0] )      # extract last time
        bias_epoch = int( re.findall(r'\d+.?\d*', last_line)[1] )   # Bias epoch
        learning_rate = float( re.findall(r'\d+.?\d*', last_line)[-1] ) # extract last learning rate
        learning_rate = 1.0e-5
    else:
        # tf_net.load_from_pytorch_net("rapid_traj_network_seed_27.pkl")
        # saver.restore(sess, save_dir + "tf_saver_seed" + ".ckpt")
        # saver.restore(sess, "./form_torch_net" + ".ckpt")
        # tf_net.build_net()
        resort_para_form_checkpoint("", sess.graph, sess, "./form_torch_net" + ".ckpt" )

        log_file = open(log_file_name, 'w+')   # New file to log
        time_bias = 0
        bias_epoch = 0   # Bias epoch
        learning_rate = 1.0e-4


    # learning_rate = 1.0e-6
    print("Log file name  = ", log_file_name)
    print("Time_bias = ", time_bias)
    print("learning_rate = ", learning_rate)
    print("bias epoch  = ", bias_epoch)
    # exit(0)

    time_train_start = cv2.getTickCount()
    # exit(0)
    # tf_out_data = sess.run(tf_net.net_output, feed_dict={tf_net.net_input: validation_input_data})
    # print("Test error = ", np.mean(tf_out_data - validation_output_data))

    # prediction = torch_policy_net(torch.from_numpy(validation_input_data).float().cuda())
    # print(prediction.shape)

    # rapid_trajectory_validate.plot_data("test")
    # rapid_trajectory_validate.output_data_for_test = prediction.data.cpu().numpy()
    # rapid_trajectory_validate.output_data = tf_out_data
    # rapid_trajectory_validate.plot_data_dash("test")
    # plt.show()

    rapid_trajectory = Rapid_trajectory_generator()
    sample_size = 20000
    # sample_size = 16000
    t_start = cv2.getTickCount()
    input_data, output_data = rapid_trajectory.load_from_file("%s/batch_%d.pkl" % (json_config["data_load_dir"], sample_size))
    # input_data, output_data = rapid_trajectory.load_from_file("%s/batch_2_%d.pkl" % (json_config["data_load_dir"], sample_size))
    # input_data, output_data = rapid_trajectory.load_from_file("%s/batch_remove_outlinear_10000.pkl" % (json_config["data_load_dir"]))
    # input_data, output_data = rapid_trajectory.load_from_file("%s/batch_remove_outlinear_25000.pkl" % (json_config["data_load_dir"]))
    # input_data, output_data = rapid_trajectory.load_from_file("%s/batch_remove_outlinear_1000.pkl" % (json_config["data_load_dir"]))
    # input_data, output_data = rapid_trajectory.load_from_file("%s/batch_remove_outlinear_1000.pkl" % (json_config["data_load_dir"]))
    # input_data, output_data = rapid_trajectory.load_from_file("%s/batch_remove_outlinear_30000.pkl" % (json_config["data_load_dir"]))
    # input_data, output_data = rapid_trajectory.load_from_file("%s/batch_30000.pkl" % (json_config["data_load_dir"]))
    # input_data, output_data = rapid_trajectory.load_from_file("%s/traj_cross_no_g.pkl" % (json_config["data_load_dir"]))
    # input_data, output_data = rapid_trajectory.load_from_file("%s/traj_cross.pkl" % (json_config["data_load_dir"]))
    # torch_input_data = torch.from_numpy(validation_input_data).float()
    # torch_output_data = torch.from_numpy(validation_output_data).float()
    if_subtract_g = 1

    # input_data = np.concatenate([input_data, validation_input_data], axis=0)
    # output_data = np.concatenate([output_data, validation_output_data], axis=0)

    # input_data = copy.deepcopy(validation_input_data)
    # output_data = copy.deepcopy(validation_output_data)


    if(if_subtract_g):
        output_data[:,8] = output_data[:,8] - 9.8
        cross_output_data[:,8] = cross_output_data[:,8] - 9.8
        validation_output_data[:,8] = validation_output_data[:,8] - 9.8
    BATCH_SIZE = int(1000 * 2)
    if(IF_DATA_ARGU==0):
        BATCH_SIZE = BATCH_SIZE*3
    BATCH_COUNT = input_data.shape[0]/BATCH_SIZE
    print("Batch size  = ", BATCH_SIZE, " count = ", BATCH_COUNT)

    # torch_input_data = torch.from_numpy(input_data).float()
    # torch_output_data = torch.from_numpy(output_data).float()
    # torch_dataset = Data.TensorDataset(torch_input_data.cpu(), torch_output_data.cpu())
    # loader = Data.DataLoader(
    #         dataset=torch_dataset,  # torch TensorDataset format
    #         batch_size=BATCH_SIZE,  # mini batch size
    #         shuffle=True,  # if shuffle the data
    #         num_workers=11,  # using how many worker.
    # )

    print("Load finish.")
    print("Set data finish.")
    # print("Batch number = ", len(loader))
    save_idx = 1000
    save_png_idx = save_idx

    # loader = Data.DataLoader(
    #     dataset=Data.TensorDataset(torch.from_numpy(input_data).float(), torch.from_numpy(output_data).float()),  # torch TensorDataset format
    #     batch_size=BATCH_SIZE,  # mini batch size
    #     shuffle=True,           # if shuffle the data
    #     num_workers=7,          # using how many worker.
    #      )
    # print("Loader_size = ", len(loader))

    t = bias_net_idx
    for epoch in range(bias_epoch, 10000000000000):
        # print("==== EPOCH %d ===="%epoch)
        learning_rate = learning_rate * 0.995
        if(learning_rate <= 1.0e-8):
            learning_rate= 1.0e-5
        for loop_1 in range(0, math.ceil(BATCH_COUNT)):
        # for loop_1 , (batch_x, batch_y) in enumerate(loader):
        #     np_data_x = batch_x.cpu().numpy()
            # np_data_y = batch_y.cpu().numpy()
            traj_index = np.random.randint(low=0, high=BATCH_COUNT - 1, size=1)[0]
            # print("traj_index = ", traj_index)
            np_data_x = input_data[range(traj_index * BATCH_SIZE, min(len(input_data) - 1, (traj_index + 1) * BATCH_SIZE)), :]
            np_data_y = output_data[range(traj_index * BATCH_SIZE, min(len(input_data) - 1, (traj_index + 1) * BATCH_SIZE)), :]
            # sample_list = random.sample(range(input_data.shape[0]), BATCH_SIZE)
            # # print(sample_list[0])
            # np_data_x = input_data[sample_list, :]
            # np_data_y = output_data[sample_list, :]

            # np_data_x = validation_input_data
            # np_data_y = validation_output_data
            # np_data_x = cross_input_data
            # np_data_y = cross_output_data
            # np_data_x = np.concatenate([np_data_x, cross_input_data], axis=0)
            # np_data_y = np.concatenate([np_data_y, cross_output_data], axis=0)
            # np_data_x = np.concatenate([np_data_x, validation_input_data], axis=0)
            # np_data_y = np.concatenate([np_data_y, validation_output_data], axis=0)
            if (IF_DATA_ARGU == 1):
                np_data_x, np_data_y = traj_augment(np_data_x, np_data_y, if_subtract_g)

            # for loop_1, (batch_x, batch_y) in enumerate(loader):
            #     np_data_x = batch_x.data.numpy()
            #     np_data_y = batch_y.data.numpy()
            loop_2_times = 1
            for loop_2 in range(loop_2_times):
                try:
                    t = t+1
                    # tf_net.train_step.run({tf_net.net_input: np_data_x, tf_net.target_output: np_data_y})
                    tf_net.train_step.run({tf_net.net_input: np_data_x, tf_net.target_output: np_data_y, tf_net.learnning_rate: learning_rate})
                    # t = epoch * math.ceil(BATCH_COUNT)  + loop_1 * loop_2_times + loop_2
                    if (t % save_idx == 0 and t != bias_net_idx):
                        try:
                            print("Save net, log_file_name = ", log_file_name)
                            save_path = saver.save(sess, "./%s/tf_saver_%d.ckpt" % (save_dir,t))
                        except:
                            # print(e)
                            pass
                        print(save_path)

                    if (t % 100 == 0):
                        # error = np.mean(sess.run(tf_net.net_output, feed_dict={tf_net.net_input: np_data_x}) - np_data_y)
                        # error = sess.run(tf_net.loss, feed_dict={tf_net.net_input: np_data_x})
                        train_error = sess.run(tf_net.loss, feed_dict={tf_net.net_input:     np_data_x,
                                                                tf_net.target_output: np_data_y})

                        valid_error_total,valid_error = sess.run([tf_net.loss, tf_net.train_loss], feed_dict={tf_net.net_input:     validation_input_data,
                                                                tf_net.target_output: validation_output_data})

                        valid_error_cross_total,valid_error_cross = sess.run([tf_net.loss, tf_net.train_loss], feed_dict={tf_net.net_input:     cross_input_data,
                                                                tf_net.target_output: cross_output_data})

                        current_train_time  = np.round((cv2.getTickCount() - time_train_start)/cv2.getTickFrequency() + time_bias,2)
                        log_str = "Time = " + str(current_train_time) + " |epoch = " + str(epoch) + ' |loop_1 = ' + str(loop_1) + ' |loop_2 = ' + str(loop_2) + ' |t = ' + str(t) + " |loss = " + str(train_error) + \
                                  " |valid_total = " + str(valid_error_total) + " |valid_train = " + str(valid_error) + \
                                  " |cross_total = " + str(valid_error_cross_total) + " |valid_cross = " + str(valid_error_cross)+ \
                                  " |learning_rate = %.10f"%learning_rate
                        print(log_str)
                        log_file.write(log_str+'\n' )
                        log_file.flush()
                    if (t % save_idx == 0 and os.path.exists("./save_png")):
                        fig = plt.figure("test")
                        plt.clf()
                        fig.set_size_inches(16 * 2, 9 * 2)
                        rapid_trajectory_validate.output_data = validation_output_data
                        rapid_trajectory_validate.plot_data("test")
                        tf_out_data = sess.run(tf_net.net_output, feed_dict={tf_net.net_input: validation_input_data})
                        rapid_trajectory_validate.output_data = tf_out_data
                        rapid_trajectory_validate.plot_data_dash("test")
                        plt.pause(0.01)
                        fig.savefig("%s/test_%d.png" % (save_dir, t))

                except:
                    print("Some error happens...")
                    pass
