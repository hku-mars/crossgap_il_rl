import numpy as np
import math, os, sys, copy
import transforms3d
import pickle
from pid_controller import Pid_controller
import tensorflow as tf

import matplotlib.pyplot as plt

DATA_DIR = "%s/pid_data"%(os.getenv('DL_data'))
print("Data save dir =  ", DATA_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Disable GPU using
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def  mkdir_soft(dir):
    try:
        os.mkdir(dir)
    except Exception as e:
        print("Make dir fail, error  =  ", e)

mkdir_soft(DATA_DIR)

data_size = int(6e6)

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

def load_test_set():
    DATA_DIR = "%s/pid_data" % (os.getenv('DL_data'))
    print("Data save dir =  ", DATA_DIR)
    pkl_file_name = "%s/real_1.pkl" % (DATA_DIR)
    data_dict_big_range = pickle.load(open(pkl_file_name, 'rb'))

    # print(data_dict_big_range)
    data_dict_big_range['in_data'] = data_dict_big_range['in_data'][:,range(12)]
    data_dict_big_range['out_data'] = data_dict_big_range['out_data'][:,range(3)]
    print("Load_data size = ", data_dict_big_range['in_data'].shape[0])
    return data_dict_big_range['in_data'], data_dict_big_range['out_data']


# In_data = [pos_err_3, spd_err_3, acc_err_3, euler_3] #12
# Out_data = [roll, pitch, yaw, thrust] #4
class Tf_pid_ctrl_net:
    def __init__(self):
        self.if_mode_rnn = 0
        self.input_size = 12
        self.output_size = 3
        self.tf_lay = []
        self.tf_w = []
        self.tf_b = []
        self.train = None
        self.net_input = None
        self.net_output = None
        self.target_output = None  # For supervised trainning
        self.weighted_val = [4, 2, 1]
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
        unit_per_lay = 40
        with tf.name_scope("Trajectory_follower_mlp_net"):
            with tf.name_scope("Lay_%s" % len(self.tf_w)):
                module = len(self.tf_w)
                self.tf_w.append(tf.Variable(tf.zeros([self.input_size, unit_per_lay]), name="layer_%d_w" % (int(module))))
                self.tf_b.append(tf.Variable(tf.zeros([unit_per_lay]), name="layer_%d_b" % (int(module))))
                # current_wb_idx = len(tf_lay_b) - 1
                self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])

                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_w[-1])
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_b[-1])
                # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.get_variable(name="layer_%d_b" % (int(module)), regularizer=self.regularizer))

            for loop in range(10):
                with tf.name_scope("Lay_%s" % len(self.tf_w)):
                    module = 2*len(self.tf_w)-1
                    self.tf_w.append(tf.Variable(tf.zeros([unit_per_lay, unit_per_lay]), name="layer_%d_w" % (int(module))))
                    self.tf_b.append(tf.Variable(tf.zeros([unit_per_lay]), name="layer_%d_b" % (int(module))))
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
                self.tf_w.append(tf.Variable(tf.zeros([unit_per_lay, self.output_size]), name="layer_%s_w" % (int(module))))
                self.tf_b.append(tf.Variable(tf.zeros([self.output_size]), name="layer_%s_b" % (int(module))))

                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_w[-1])
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.tf_b[-1])

                self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])
        self.net_output = tf.identity(self.tf_lay[-1], name = "Trajectory_follower_mlp_net_out" )
        return self.tf_lay

    def build_net(self, lay_input = None):

        self.target_output = tf.placeholder(tf.float32, [None, self.output_size], name='target_output');
        self.learnning_rate = tf.placeholder(tf.float32, shape=[])
        self.tf_lay = []
        self.tf_w = []
        self.tf_b = []
        self.rnn_hidden_state_in = []
        self.rnn_hidden_state_out = []
        self.regularizer = tf.contrib.layers.l1_regularizer(1e-7)
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
        import torch
        self.learnning_rate = tf.placeholder(tf.float32, shape=[])
        # logger.writelines("run torch_net_to_tf_net_mlp")
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
        import torch
        # logger.writelines("run torch_net_to_tf_net")
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

    def euler_to_ration_matrx(self, euler_x, euler_y):
        # MAX_ANGLE = math.pi/2
        # euler_x = tf.maximum(euler_x, -MAX_ANGLE)
        # euler_x = tf.minimum(euler_x, MAX_ANGLE)
        # euler_y = tf.maximum(euler_y, -MAX_ANGLE)
        # euler_y = tf.minimum(euler_y, MAX_ANGLE)
        cos_rot_x = tf.cos(euler_x)
        cos_rot_y = tf.cos(euler_y)
        # cos_rot_z = tf.cos(euler_z)

        sin_rot_x = tf.sin(euler_x)
        sin_rot_y = tf.sin(euler_y)
        # sin_rot_z = tf.sin(euler_z)

        one = tf.ones_like(cos_rot_x, dtype=tf.float32)
        zero = tf.zeros_like(cos_rot_x, dtype=tf.float32)

        print("Input vec shape = ", one._shape)

        rot_x = tf.stack([tf.concat([one, zero, zero], axis=1),
                          tf.concat([zero, cos_rot_x, sin_rot_x], axis=1),
                          tf.concat([zero, -sin_rot_x, cos_rot_x], axis=1)], axis=1)

        rot_y = tf.stack([tf.concat([cos_rot_y, zero, -sin_rot_y], axis=1),
                          tf.concat([zero, one, zero], axis=1),
                          tf.concat([sin_rot_y, zero, cos_rot_y], axis=1)], axis=1)

        rot_z = tf.stack([tf.concat([one, zero, zero], axis=1),
                          tf.concat([zero, one, zero], axis=1),
                          tf.concat([zero, zero, one], axis=1)], axis=1)

        rot_matrix = tf.matmul(rot_x, tf.matmul(rot_y, rot_z))
        print("Rotation matrix shape = ", rot_matrix._shape)

        return rot_matrix

    def get_axis_angle(self, rot_matrix):
        # From http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/
        m_00 = tf.slice(rot_matrix, [0, 0, 0], [-1, 1, 1])
        m_11 = tf.slice(rot_matrix, [0, 1, 1], [-1, 1, 1])
        m_22 = tf.slice(rot_matrix, [0, 2, 2], [-1, 1, 1])
        angle = tf.reshape( tf.acos((m_00 + m_11 + m_22 - 1.0)/2.0), shape=[-1,1])
        self.m_00 = m_00
        self.m_11 = m_11
        self.m_22 = m_22
        return angle


    def rotation_matrix_mul_100(self, rot_mat):
        one = tf.reshape(tf.ones_like(tf.slice(rot_mat, begin=[0,0,0], size = [-1,1,1] ) , dtype=tf.float32), shape=[-1,1])
        zero = tf.reshape(tf.zeros_like(tf.slice(rot_mat, begin=[0,0,0], size = [-1,1,1]), dtype=tf.float32), shape=[-1,1])
        print("One shape = ", one._shape)

        vec = tf.reshape(tf.stack(tf.concat( [one, zero,zero ], axis=1) ), shape = [-1,3,1])

        print("Vec shape = ", vec._shape)
        rot_mul_vec = tf.reshape(tf.matmul(rot_mat, vec), shape = [-1,3])
        print("Rot_mul_vec shape = ", rot_mul_vec._shape)

        return rot_mul_vec


    def rotation_matrix_mul_010(self, rot_mat):
        one = tf.reshape(tf.ones_like(tf.slice(rot_mat, begin=[0,0,0], size = [-1,1,1] ) , dtype=tf.float32), shape=[-1,1])
        zero = tf.reshape(tf.zeros_like(tf.slice(rot_mat, begin=[0,0,0], size = [-1,1,1]), dtype=tf.float32), shape=[-1,1])
        print("One shape = ", one._shape)

        vec = tf.reshape(tf.stack(tf.concat( [ zero,one,zero ], axis=1) ), shape = [-1,3,1])

        print("Vec shape = ", vec._shape)
        rot_mul_vec = tf.reshape(tf.matmul(rot_mat, vec), shape = [-1,3])
        print("Rot_mul_vec shape = ", rot_mul_vec._shape)

        return rot_mul_vec

    def rotation_matrix_mul_001(self, rot_mat):
        one = tf.reshape(tf.ones_like(tf.slice(rot_mat, begin=[0,0,0], size = [-1,1,1] ) , dtype=tf.float32), shape=[-1,1])
        zero = tf.reshape(tf.zeros_like(tf.slice(rot_mat, begin=[0,0,0], size = [-1,1,1]), dtype=tf.float32), shape=[-1,1])
        print("One shape = ", one._shape)

        vec = tf.reshape(tf.stack(tf.concat( [zero,zero, one], axis=1) ), shape = [-1,3,1])

        print("Vec shape = ", vec._shape)
        rot_mul_vec = tf.reshape(tf.matmul(rot_mat, vec), shape = [-1,3])
        print("Rot_mul_vec shape = ", rot_mul_vec._shape)

        return rot_mul_vec

    def compate_euler(self, net_roll,net_pitch, tar_roll, tar_pitch):
        rot_x = tf.sin(net_pitch) - tf.sin(tar_pitch)
        rot_y = tf.cos(net_pitch)*tf.sin(net_roll) - tf.cos(tar_pitch)*tf.sin(tar_roll)
        rot_z = tf.cos(net_roll)*tf.sin(net_pitch) - tf.cos(tar_roll)*tf.sin(tar_pitch)
        return tf.sqrt(tf.pow(rot_x,2)+tf.pow(rot_y,2) + tf.pow(rot_y,2))

    def compare_euler_angle_error(self, net_roll,net_pitch, tar_roll, tar_pitch):
        cx1 = tf.cos(net_roll)
        sx1 = tf.sin(net_roll)
        cy1 = tf.cos(net_pitch)
        sy1 = tf.sin(net_pitch)

        cx2 = tf.cos(tar_roll)
        sx2 = tf.sin(tar_roll)
        cy2 = tf.cos(tar_pitch)
        sy2 = tf.sin(tar_pitch)

        m00 = cy1*cy2 + sy1*sy2
        m11 = cx1*cx2 + cy1*cy2*sx1*sx2 + sx1*sx2*sy1*sy2
        m22 = sx1*sx2 + cx1*cx2*sy1*sy2 + cx1*cx2*cy1*cy2
        self.acos =  ( m00 + m11 + m22 - 1)/2.0 * 0.99999
        return tf.acos(self.acos)

    def train_loop(self):
        print(self.net_output, self.target_output)
        diff = self.net_output - self.target_output

        # angle_err = tf.slice(diff, [0, 0], [-1, 3])
        # yaw_rr = tf.slice(diff, [0, 2], [-1, 1])
        # thr_err = tf.slice(diff, [0, 3], [-1, 1])

        # angle_err = tf.norm(tf.slice(diff, [0, 0], [-1, 2]), axis=1, keepdims=True )
        # angle_err = tf.pow(tf.slice(diff, [0, 0], [-1, 1]),2) +  tf.pow(tf.slice(diff, [0, 1], [-1, 1]),2)

        # Loss function : http://ncfrn.mcgill.ca/members/pubs/AtAllCosts_mactavish_crv15.pdf

        angle_err = tf.losses.huber_loss(labels=tf.slice(self.net_output, [0, 0], [-1, 1]) * 57.3, predictions=tf.slice(self.target_output, [0, 0], [-1, 1]) * 57.3, delta=5.0) + \
                tf.losses.huber_loss(labels=tf.slice(self.net_output, [0, 1], [-1, 1]) * 57.3, predictions=tf.slice(self.target_output, [0, 1], [-1, 1]) * 57.3, delta=5.0)
        # angle_err = tf.losses.huber_loss(labels=tf.slice(self.net_output, [0, 0], [-1, 1]) * 57.3, predictions=tf.slice(self.target_output, [0, 0], [-1, 1]) * 57.3, delta=5.0) + \
        thr_err = tf.losses.huber_loss( labels=tf.slice(self.net_output, [0, 2], [-1, 1])*2, predictions= tf.slice(self.target_output, [0, 2], [-1, 1])*2 , delta= 1.0)

        net_roll = tf.slice(self.net_output, [0, 0], [-1, 1])
        net_pitch = tf.slice(self.net_output, [0, 1], [-1, 1])
        net_yaw = tf.zeros_like(tf.slice(self.net_output, [0, 0], [-1, 1]))

        tar_roll = tf.slice(self.target_output, [0, 0], [-1, 1])
        tar_pitch = tf.slice(self.target_output, [0, 1], [-1, 1])
        tar_yaw = tf.zeros_like(tf.slice(self.target_output, [0, 0], [-1, 1]))

        self.net_rot_mat = self.euler_to_ration_matrx(net_roll, net_pitch)
        self.tar_rot_mat = self.euler_to_ration_matrx(tar_roll, tar_pitch)
        self.net_roll = net_roll
        self.net_pitch = net_pitch
        if(0):
            if(0):
                self.tar_rot_mat_inv = tf.transpose(self.tar_rot_mat, perm=[0,2,1])
                self.rotation_mat_diff = tf.matmul(net_rot_mat,self.tar_rot_mat_inv)
                self.rotation_diff = self.get_axis_angle( self.rotation_mat_diff )
                print("Rot_diff shape = ",  self.rotation_diff._shape)
            else:
                if_all_vec = 0

                net_vec_001 = self.rotation_matrix_mul_001(self.net_rot_mat)
                tar_vec_001 = self.rotation_matrix_mul_001(self.tar_rot_mat)

                if(if_all_vec):
                    net_vec_100 = self.rotation_matrix_mul_100(self.net_rot_mat)
                    tar_vec_100 = self.rotation_matrix_mul_100(self.tar_rot_mat)

                    net_vec_010 = self.rotation_matrix_mul_010(self.net_rot_mat)
                    tar_vec_010 = self.rotation_matrix_mul_010(self.tar_rot_mat)
                    self.net_vec_100 = net_vec_100
                    self.net_vec_010 = net_vec_010
                    self.net_vec_001 = net_vec_001
                if (if_all_vec):
                    self.rotation_diff = tf.norm(tf.abs(net_vec_001 - tar_vec_001), axis=1, keep_dims=True) + tf.norm(tf.abs(net_vec_100 - tar_vec_100), axis=1, keep_dims=True) + \
                                         tf.norm(tf.abs(net_vec_010 - tar_vec_010), axis=1, keep_dims=True)
                    # self.rotation_diff =  tf.maximum(tf.maximum(
                    #         tf.norm(net_vec_001 - tar_vec_001, axis=1, keep_dims=True),
                    #         tf.norm(net_vec_100 - tar_vec_100, axis=1, keep_dims=True)),
                    #         tf.norm(net_vec_010 - tar_vec_010, axis=1, keep_dims=True))
                else:
                    self.rotation_diff = tf.norm(net_vec_001 - tar_vec_001, axis=1, keep_dims=True)
                print("Diff shape = ", (net_vec_001 - tar_vec_001)._shape)
        else:
            # self.rotation_diff = self.compate_euler(net_roll, net_pitch, tar_roll, tar_pitch)
            self.rotation_diff = self.compare_euler_angle_error(net_roll, net_pitch, tar_roll, tar_pitch)

        print("Ratation_diff shape =  ", self.rotation_diff._shape)
        print("Thr shape = ", thr_err._shape)
        print("Euler_angle shape = ", angle_err._shape)


        # for weight in self.tf_w:
        #     weight_temp = tf.get_variable(weight, regularizer=self.regularizer)
        #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_temp)

        reg_variables =  tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # self.train_loss = tf.reduce_mean(tf.abs(angle_err)+tf.abs(yaw_rr)*0.1+tf.abs(thr_err )*5 , name="loss")

        # self.train_loss = tf.reduce_mean(angle_err*0.5+ tf.abs(thr_err ) , name="loss")
        self.train_loss = tf.reduce_mean(self.rotation_diff*57.3 + thr_err  , name="loss")

        # print(self.regularizer)
        # print(reg_variables)
        # self.total_loss = self.train_loss
        self.total_loss = self.train_loss + tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)

        self.train_step = tf.train.AdamOptimizer(self.learnning_rate).minimize(self.total_loss)
        # self.train_step = tf.train.GradientDescentOptimizer(self.learnning_rate).minimize(self.total_loss)


if __name__ == '__main__':
    print("Hello, this is tf_pid_network")
    if_mode_rnn = 0


    traj_follower = Tf_pid_ctrl_net()
    if(0):
        import torch
        net = torch.load('demo_network_3700000.pkl')
        print(net)
        traj_follower.torch_net_to_tf_net_mlp(net)
    else:
        traj_follower.build_net()
    traj_follower.train_loop()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    if(if_mode_rnn):
        tf.summary.FileWriter("pid_rnn_tf_log/", sess.graph)
    else:
        tf.summary.FileWriter("pid_mlp_tf_log_temp/", sess.graph)

    if (if_mode_rnn):
        net_save_dir = "./tf_pid_rnn_net"
    else:
        net_save_dir = "./tf_pid_mlp_net"

    mkdir_soft(net_save_dir)
    saver = tf.train.Saver(max_to_keep=50000)



    # if(if_mode_rnn):
    #     # save_path = saver.save(sess, "%s/save_net_rnn.ckpt"%net_save_dir)
    #     saver.restore(sess,"%s/save_net_rnn.ckpt"%net_save_dir)
    # else:
    #     # save_path = saver.save(sess, "%s/save_net_mlp.ckpt"%net_save_dir)
    #     saver.restore(sess,"%s/save_net_mlp.ckpt"%net_save_dir)
    print("Build net finish")

    import torch
    import torch.utils.data as Data

    input_data_for_train, output_data_for_train = load_pid_data()
    input_data_for_test, output_data_for_test = load_test_set()

    # input_data_for_train = input_data_for_test
    # output_data_for_train = output_data_for_test

    for _ in range(10):
        input_data_for_train = np.concatenate([input_data_for_train, input_data_for_test], axis=0)
        output_data_for_train = np.concatenate([output_data_for_train, output_data_for_test], axis=0)

    # input_data_for_train[:, range(2)] = input_data_for_train[:, range(3)] *57.3
    # output_data_for_train[:, range(3)] = output_data_for_train[:, range(3)] *57.3

    # input_data_for_test = input_data_for_train[range(100), :]
    # output_data_for_test =output_data_for_train[range(100), :]

    # BATCH_SIZE = int(1500 * 1.3 * 100)
    BATCH_SIZE = 1000
    BATCH_COUNT = input_data_for_train.shape[0] / BATCH_SIZE
    print("Batch size  = ", BATCH_SIZE, " count = ", BATCH_COUNT)

    save_idx = 1e4
    show_idx = 1e2
    Learnning_rate = 5e-5

    bias_net_idx = 1170000
    bias_net_idx = 1870000
    bias_net_idx = 2020000  # loop 2 = 1000
    bias_net_idx = 3060000
    bias_net_idx = 15630000 # before cancel limitation
    bias_net_idx = 16530000
    bias_net_idx = 12780000 # change network, converge
    bias_net_idx = 23540000 #
    bias_net_idx = 24010000
    bias_net_idx = 26600000
    bias_net_idx = 26310000 # before add weight decay
    bias_net_idx = 28600000 # add sample decrease to 5%
    bias_net_idx = 34410000 # add sample decrease to 5%
    bias_net_idx = 49770000 # 0.455
    bias_net_idx = 54120000
    bias_net_idx = 116000000 # last train  from ali ware
    bias_net_idx = 118300000
    bias_net_idx = 121640000
    bias_net_idx = 124244000
    bias_net_idx = 124580000
    bias_net_idx = 124920000
    bias_net_idx = 125240000
    bias_net_idx = 126110000
    bias_net_idx = 126310000
    bias_net_idx = 127980000 # 0 -> 0.01
    bias_net_idx = 128300000
    bias_net_idx = 128650000 # +- 90
    bias_net_idx = 128690000
    bias_net_idx = 128795000
    bias_net_idx = 129100000 # rotation angle diff as cost
    bias_net_idx = 129700000 # small set
    bias_net_idx = 129870000
    bias_net_idx = 130250000
    bias_net_idx = 130330000
    # bias_net_idx = 128670000
    # bias_net_idx = 128690000
    # bias_net_idx = 128390000
    # bias_net_idx = 26610000
    # bias_net_idx = 36310000
    t = bias_net_idx

    if(bias_net_idx != 0):
        saver.restore(sess,  "%s/tf_saver_%d.ckpt" % (net_save_dir,bias_net_idx))

    for epoch in range(100000000000000000):

        Learnning_rate =  Learnning_rate *0.95
        if(Learnning_rate < 5e-10000):
            Learnning_rate = 1e-5

        BATCH_SIZE = int(BATCH_SIZE + 1  )
        if(BATCH_SIZE > 300000):
            BATCH_SIZE = 1000
        BATCH_COUNT = input_data_for_train.shape[0] / BATCH_SIZE
        print("Learning rate = ", Learnning_rate , ", BATCH_SIZE = ", BATCH_SIZE)

        sample_idxs = np.random.randint(low = input_data_for_test.shape[0] - 1000, high=input_data_for_test.shape[0] - 1, size=int(BATCH_SIZE*0.00)).ravel().tolist()
        add_sample_x = input_data_for_test[sample_idxs , :]
        add_sample_y = output_data_for_test[sample_idxs, :]

        for loop_1 in range(0, math.ceil(BATCH_COUNT)):

            traj_index = np.random.randint(low=0, high=BATCH_COUNT - 1, size=1)[0]
            # print("traj_index = ", traj_index)
            np_data_x = input_data_for_train[range(traj_index * BATCH_SIZE, min(len(input_data_for_train) - 1, (traj_index + 1) * BATCH_SIZE)), :]
            np_data_y = output_data_for_train[range(traj_index * BATCH_SIZE, min(len(input_data_for_train) - 1, (traj_index + 1) * BATCH_SIZE)), :]

            np_data_x = np.concatenate([np_data_x, add_sample_x], axis=0)
            np_data_y = np.concatenate([np_data_y, add_sample_y], axis=0)

            loop_2_times = 1

            for loop_2 in range(loop_2_times):

                if(0):
                    dbg_rot, dbg_rot_diff,rot_diff, rot_mat_diff = sess.run([traj_follower.tar_rot_mat, traj_follower.tar_rot_mat_inv,
                                                               traj_follower.rotation_diff ,traj_follower.rotation_mat_diff],
                                                              feed_dict={traj_follower.net_input:     np.matrix(np_data_x),
                                                                         traj_follower.target_output: np.matrix(np_data_y)})
                    m_00,m_11,m_22 = sess.run([traj_follower.m_00, traj_follower.m_11,traj_follower.m_22] ,
                                              feed_dict={traj_follower.net_input:     np.matrix(np_data_x),
                                                                         traj_follower.target_output: np.matrix(np_data_y)})
                    print(dbg_rot, '\n=====\n', dbg_rot_diff, '\n=++++=\n', rot_diff , "\n-----\n ", rot_mat_diff)
                    print("Mat_m = ", m_00," --- ", m_11," --- ", m_22)

                    exit(0)
                t = t+1
                if(0):
                    rotdiff, test_x, test_y = sess.run( [traj_follower.acos, traj_follower.net_roll,traj_follower.net_pitch ], feed_dict={traj_follower.net_input:      np.matrix(np_data_x),
                                                  traj_follower.target_output:  np.matrix(np_data_y),
                                                  traj_follower.learnning_rate: Learnning_rate} )
                    print( np.min(rotdiff), np.max(rotdiff), np.min(np.abs(rotdiff)) , ' --- ',  np.min(test_x), np.max(test_x), np.min(np.abs(test_x)),
                           ' --- ', np.min(test_y), np.max(test_y), np.min(np.abs(test_y)),)

                traj_follower.train_step.run({traj_follower.net_input:      np.matrix(np_data_x),
                                              traj_follower.target_output:  np.matrix(np_data_y),
                                              traj_follower.learnning_rate: Learnning_rate})
                # print(t)
                if ((t % save_idx == 0) and t != bias_net_idx):

                    try:
                        save_path = saver.save(sess, "%s/tf_saver_%d.ckpt" % (net_save_dir,t))
                        print(save_path)
                    except Exception as e:
                        print("Save net errror")
                        print(e)

                if (t % (show_idx*10) == 0):

                    loss, loss_total, train_angle_diff = sess.run([traj_follower.train_loss, traj_follower.total_loss, traj_follower.rotation_diff],
                                                                  feed_dict={traj_follower.net_input:     np.matrix(np_data_x),
                                                                             traj_follower.target_output: np.matrix(np_data_y)})
                    loss_test, angle_diff = sess.run([traj_follower.train_loss, traj_follower.rotation_diff], feed_dict={traj_follower.net_input:     np.matrix(input_data_for_test),
                                                                                                                         traj_follower.target_output: np.matrix(output_data_for_test)})
                    # error_test = traj_follower.train_loss.run({traj_follower.net_input: np.matrix(input_data_for_test), traj_follower.target_output: np.matrix(output_data_for_test)})


                    error = np.mean(np.abs(sess.run(traj_follower.net_output, feed_dict={traj_follower.net_input: np_data_x}) - np_data_y))
                    net_test_out = sess.run(traj_follower.net_output, feed_dict={traj_follower.net_input: np.matrix(input_data_for_test)})
                    error_test = np.mean(np.abs(net_test_out - output_data_for_test))
                    log_str = "epoch = " + str(epoch) + ' |loop_1 = ' + str(loop_1) + ' |loop_2 = ' + str(loop_2) + ' |t =' + str(t) \
                              + " |loss = " + "%.3f" % loss + " |loss_total = " + "%.3f" % loss_total  + "|angle_diff = " + str(np.round(np.mean(np.abs(train_angle_diff)), 9)) \
                              + " |error= " + "%.3f" % error + " |errot_test = " + "%.3f" % error_test \
                              +" |angle_diff= "  + str(np.round(np.mean(np.abs(angle_diff)), 9))

                    print(log_str)

                    if (os.path.exists("./save_png")):
                        plt.figure('compare')
                        plt.cla()
                        plt.plot(output_data_for_test[:, 0]*57.3, 'r--', label='target_roll')
                        plt.plot(output_data_for_test[:, 1]*57.3, 'b--', label='target_pitch')
                        plt.plot(output_data_for_test[:, 2], 'k--', label='target_roll')

                        plt.plot(net_test_out[:, 0]*57.3, 'r-', label='target_roll')
                        plt.plot(net_test_out[:, 1]*57.3, 'b-', label='target_pitch')
                        plt.plot(net_test_out[:, 2], 'k-', label='target_roll')
                        plt.grid('on')
                        plt.title("Mean_error = "+str(error_test))
                        plt.legend()
                        plt.pause(0.1)


                        # if (t % save_idx == 0):
                #     rapid_trajectory_validate.output_data = validation_output_data
                #     rapid_trajectory_validate.plot_data("test")
                #     tf_out_data = sess.run(tf_net.net_output, feed_dict={tf_net.net_input: validation_input_data})
                #     rapid_trajectory_validate.output_data = tf_out_data
                #     rapid_trajectory_validate.plot_data_dash("test")
                #     fig.savefig("%s/test_%d.png" % (save_dir, t))
