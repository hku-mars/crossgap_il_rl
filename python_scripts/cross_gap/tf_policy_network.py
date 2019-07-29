import tensorflow as tf
import numpy as np
from tensorflow.contrib import graph_editor as ge
import os, copy, sys
import re
sys.path.append("..\\")

# from deep_drone.tf_trajectory_follower import Trajectory_follower_tf
from cross_gap.tools.tf_pid_network import Tf_pid_ctrl_net as Trajectory_follower_tf
from deep_drone.tf_rapid_trajectory import Plannning_net_tf
from tensorflow.python import pywrap_tensorflow
IF_RL = 0
def resort_para_form_checkpoint( prefix, graph, sess):
    # ckpt_name_vec = ["./tf_net/planning_net/tf_saver_252750.ckpt", "./tf_net/control_mlp_net_train/tf_saver_2318100.ckpt"]
    # ckpt_name_vec = ["./tf_net/planning_net/tf_saver_252750.ckpt", "./tf_net/control_mlp_net_train/tf_saver_1300000.ckpt"]
    ckpt_name_vec = ["./tf_net/planning_net/tf_saver_107840000.ckpt", "./tf_net/pid_net/tf_saver_109330000.ckpt"]
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
                try:
                    tensor = graph.get_tensor_by_name(key)
                    sess.run(tf.assign(tensor, reader.get_tensor(_key)))
                    # print(tensor)
                except Exception as e:
                    # print(key, " can not be restored, e= ",str(e))
                    pass

def return_notation_list(str, pattern):
    return  [m.start() for m in re.finditer(pattern, str)]

class Policy_network():

    def resore_form_rl_net(self,ckpt_name, graph, sess):
        print("Restore form RL net")

        print("===== Prase data from %s =====" % ckpt_name)
        net_prefix = 'pi/pi'
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for _key in var_to_shape_map:
            print(_key)
            # print("tensor_name: ", key)
            # print(reader.get_tensor(key))
            # tensor = graph.get_tensor_by_name(key)
            if (str(_key).startswith('%s/net/'%net_prefix) or
                str(_key).startswith('%s/Trajectory_follower_mlp_net/'%net_prefix)):
                notaion_list =  [m.start() for m in re.finditer('/', _key)]
                key = _key[int(notaion_list[1]+1):len(_key)]+ ":0"
                # print(key)
                try:
                    tensor = graph.get_tensor_by_name(key)
                    sess.run(tf.assign(tensor, reader.get_tensor(_key)))
                    # print(tensor)
                except Exception as e:
                    print(key, " can not be restored, e= ",str(e))
                    pass


    def save_graph(self, graph):
        self.graph_dir = "./quadrotor_control_graph/"
        log_dir = self.graph_dir
        try:
            os.mkdir(log_dir)
        except:
            pass
        print("save graph to ", log_dir)
        tf.summary.FileWriter(log_dir, graph)

    def prase_checkpoint_data(self, checkpoint_name):
        print("===== Prase data from %s =====" % checkpoint_name)
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print("tensor_name: ", key)
            # print(reader.get_tensor(key))

    def resort_para_form_checkpoint(self, _ckpt_name_vec, graph, sess):
        # with tf.name_scope("restore"):
        if( isinstance(_ckpt_name_vec, list)):
            ckpt_name_vec = _ckpt_name_vec
        else:
            ckpt_name_vec = [_ckpt_name_vec]

        with tf.name_scope ("restore"):
            for ckpt_name in ckpt_name_vec:
                print("===== Restore data from %s =====" % ckpt_name)
                reader = pywrap_tensorflow.NewCheckpointReader(ckpt_name)
                var_to_shape_map = reader.get_variable_to_shape_map()
                for key in var_to_shape_map:
                    # print("tensor_name: ", key)
                    # print(reader.get_tensor(key))
                    # tensor = graph.get_tensor_by_name(key)
                    try:
                        tensor = graph.get_tensor_by_name(key + ":0")
                        sess.run(tf.assign(tensor, reader.get_tensor(key)))
                        # print(tensor)
                    except:
                        # print(key, " can not be restored")
                        pass

    def list_all_var(self):
        variables = tf.contrib.framework.get_variables_to_restore(include=[""])
        for v in variables:
            print("--> ", v.name)

    def load_para(self):
        self.resort_para_form_checkpoint([control_net_ckpt_name, planning_net_ckpt_name], self.sess.graph, self.sess)

    def __init__(self, X=None, if_restore = 1): # if X is not none, X is the observation shape = [noen, 29]

        if X is not None:
            self.observation =X

        self.control_network = Trajectory_follower_tf()
        self.plan_network = Plannning_net_tf()
        self.if_mode_rnn = self.control_network.if_mode_rnn

        if (sys.platform == 'win32'):
            # planning_net_ckpt_name = "%s/tf_net/planning_net/tf_saver_250010.ckpt" % ("../../networks")
            # planning_net_ckpt_name = "%s/tf_net/planning_net/tf_saver_252750.ckpt" %("../../networks")
            # planning_net_ckpt_name = "%s/tf_net/planning_net/tf_saver_8262000.ckpt"% ("../../networks")
            planning_net_ckpt_name = "%s/tf_net/planning_net/tf_saver_107840000.ckpt"% ("../../networks")
        else:
            planning_net_ckpt_name = "%s/catkin_ws/src/cross_gap/script/tf_net/planning_net/tf_saver_34000000.ckpt" % os.getenv("HOME")
            planning_net_ckpt_name = "%s/catkin_ws/src/cross_gap/script/tf_net/planning_net/tf_saver_107840000.ckpt" % os.getenv("HOME")

        # planning_net_ckpt_name = "./tf_net/planning_net/save_net_new.ckpt"
        if(self.if_mode_rnn):
            control_net_ckpt_name = "%s/tf_net/control_rnn_net/save_net_rnn.ckpt"% ("../../networks")
        else:
            if (sys.platform == 'win32'):
                # control_net_ckpt_name = "%s/tf_net/control_mlp_net/save_net_mlp.ckpt"%("../../networks")
                control_net_ckpt_name = "%s/tf_net/pid_net/tf_saver_60150000.ckpt"%("../../networks")
                control_net_ckpt_name = "%s/tf_net/pid_net/tf_saver_93720000.ckpt"%("../../networks")
                control_net_ckpt_name = "%s/tf_net/pid_net/tf_saver_109330000.ckpt"% ("../../networks")
            else:
                control_net_ckpt_name = "%s/catkin_ws/src/cross_gap/script/tf_net/control_mlp_net/save_net_mlp.ckpt" % os.getenv("HOME")
                control_net_ckpt_name = "%s/catkin_ws/src/cross_gap/script/tf_net/pid_net/tf_saver_60150000.ckpt" % os.getenv("HOME")
                control_net_ckpt_name = "%s/catkin_ws/src/cross_gap/script/tf_net/pid_net/tf_saver_93720000.ckpt" % os.getenv("HOME")
                control_net_ckpt_name = "%s/catkin_ws/src/cross_gap/script/tf_net/pid_net/tf_saver_109330000.ckpt" % os.getenv("HOME")
            # control_net_ckpt_name = "./tf_net/control_mlp_net_train/tf_saver_20000.ckpt"
            # control_net_ckpt_name = "./tf_net/control_mlp_net_train/tf_saver_2318100.ckpt"

        self.if_save_log = 0


        # self.graph_planning = tf.Graph()
        # with self.graph_planning.as_default():
        self.graph_planning = tf.get_default_graph()
        with tf.get_default_graph().as_default():
            print("Build planning net")
            if X is None:
                self.plan_network.build_net()
            else:
                self.plan_network.build_net(tf.slice(X, [0, 12], [-1, 17]))

            self.plan_network.train_loop()
            # self.plan_network.train_loop()

            with tf.name_scope("Current_plane_state"):
                if X is None:
                    self.net_plane_state = tf.placeholder_with_default(np.zeros([1, 12], dtype=np.float32), shape=[None,12], name = "Plane_state")
                else:
                    # self.net_plane_state = tf.placeholder_with_default(tf.slice(X, [0, 0], [-1, 9]), shape=[None, 9], name="Plane_state") , name="gravity")
                    # self.net_plane_state = tf.placeholder_with_default(tf.slice(X, [0, 0], [-1, 12])+ tf.constant([0, 0, 0,0, 0, 0, 0, 0, 9.8]) , shape=[None, 12], name="Plane_state")
                    self.net_plane_state = tf.placeholder_with_default(tf.slice(X, [0, 0], [-1, 12]), shape=[None, 12], name="Plane_state")

            print("Build control net")
            self.control_net_input = tf.concat([self.plan_network.tf_lay[-1] - tf.slice(self.net_plane_state, [0, 0], [-1, 9]) ,
                                                tf.slice(self.net_plane_state, [0, 9], [-1, 3])], 1)
            self.control_network.build_net(self.control_net_input)

            # with tf.name_scope("ctrl_network"):
            # collect_plan = self.graph_planning.get_collection("plan_network/")
            if(0):
                self.sess = tf.InteractiveSession(graph=self.graph_planning)
            else:
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.sess = tf.InteractiveSession(graph=self.graph_planning, config=config)
            self.sess.run(tf.global_variables_initializer())
            if (if_restore):
                if(IF_RL):
                    if (sys.platform == 'win32'):
                        ckpt_name = '%s/tf_net/rl_net/seed' % ("../../networks")
                        ckpt_name = '%s/tf_net/rl_net/iter_1180'% ("../../networks")
                        ckpt_name = '%s/tf_net/rl_net/iter_2460'% ("../../networks")
                        ckpt_name = '%s/tf_net/rl_net/iter_640'% ("../../networks")
                        ckpt_name = '%s/tf_net/rl_net/iter_980'% ("../../networks")
                    else:
                        ckpt_name = '%s/catkin_ws/src/cross_gap/script/tf_net/rl_net/iter_1180'% os.getenv("HOME")
                    self.resore_form_rl_net(ckpt_name, self.sess.graph, self.sess)
                else:
                    self.resort_para_form_checkpoint([control_net_ckpt_name,planning_net_ckpt_name], self.sess.graph, self.sess)

        # hidden state
        if(self.if_mode_rnn):
            self.feed_hidden_state_0 = np.zeros(self.control_network.rnn_hidden_state_in[0].shape)
            self.feed_hidden_state_1 = np.zeros(self.control_network.rnn_hidden_state_in[1].shape)

        # predefine network vector
        self.nn_input_vec = np.zeros([1, 12])
        self.nn_output_vec = np.zeros([1, 4])
        self.save_graph(self.sess.graph)

        # https://gist.github.com/marta-sd/ba47a9626ae2dbcc47094c196669fd59
        print("----- finish merge -----")
        # return self.control_network.net_output

    def  planning_summary(self, pos_error, spd_error, acc_error, tf_out_data):
        pos_error_norm = np.linalg.norm(pos_error, keepdims=1, axis=1)
        spd_error_norm = np.linalg.norm(spd_error, keepdims=1, axis=1)
        acc_error_norm = np.linalg.norm(acc_error, keepdims=1, axis=1)
        print("==== Traj compartion ====")
        print('Pos error [Max, Mean, Min] = [%.3f, %.3f, %.3f] ' % (np.max(pos_error_norm), np.mean(pos_error_norm), np.min(pos_error_norm)))
        print('Spd error [Max, Mean, Min] = [%.3f, %.3f, %.3f] ' % (np.max(spd_error_norm), np.mean(spd_error_norm), np.min(spd_error_norm)))
        print('Acc error [Max, Mean, Min] = [%.3f, %.3f, %.3f] ' % (np.max(acc_error_norm), np.mean(acc_error_norm), np.min(acc_error_norm)))

        print('===== Acc =====')
        print('Acc_range_x = [ Max, Min ] = [ %.3f, %.3f ]' % (np.max(tf_out_data[:, 6]), np.min(tf_out_data[:, 6])))
        print('Acc_range_y = [ Max, Min ] = [ %.3f, %.3f ]' % (np.max(tf_out_data[:, 7]), np.min(tf_out_data[:, 7])))
        print('Acc_range_z = [ Max, Min ] = [ %.3f, %.3f ]' % (np.max(tf_out_data[:, 8]), np.min(tf_out_data[:, 8])))

        print('===== Spd =====')
        print('Spd_range_x = [ Max, Min ] = [ %.3f, %.3f ]' % (np.max(tf_out_data[:, 3]), np.min(tf_out_data[:, 3])))
        print('Spd_range_y = [ Max, Min ] = [ %.3f, %.3f ]' % (np.max(tf_out_data[:, 4]), np.min(tf_out_data[:, 4])))
        print('Spd_range_z = [ Max, Min ] = [ %.3f, %.3f ]' % (np.max(tf_out_data[:, 5]), np.min(tf_out_data[:, 5])))

        print('===== Pos =====')
        print('Pos_range_x = [ Max, Min ] = [ %.3f, %.3f ]' % (np.max(tf_out_data[:, 0]), np.min(tf_out_data[:, 0])))
        print('Pos_range_y = [ Max, Min ] = [ %.3f, %.3f ]' % (np.max(tf_out_data[:, 1]), np.min(tf_out_data[:, 1])))
        print('Pos_range_z = [ Max, Min ] = [ %.3f, %.3f ]' % (np.max(tf_out_data[:, 2]), np.min(tf_out_data[:, 2])))

        print('===== End state error =====')
        print('End pos error  = ', np.round(pos_error[-1], 3))
        print('End spd error  = ', np.round(spd_error[-1], 3))
        print('End acc error  = ', np.round(acc_error[-1], 3))

    def compare_traditional_traj_and_network(self, in_narrow_gap, if_display = 1, resolution = 1000):

        pos_torlerance = 0.05
        sys.path.append("%s" % os.getenv("CROSS_GAP_WORK_DIR"))
        # from narrow_gap import narrow_gap
        from Rapid_trajectory_generator import Rapid_trajectory_generator
        import matplotlib.pyplot as plt
        rapid_trajectory = Rapid_trajectory_generator()
        rapid_trajectory.generate_trajectory_np_array(np.array(in_narrow_gap.rapid_nn_input_vector.T[0]))
        if (resolution != 1000):
            rapid_trajectory.export_data(resolution, if_random=0)
        rapid_trajectory.output_data[:, 8] = rapid_trajectory.output_data[:, 8] - 9.8


        tf_out_data = self.sess.run(self.plan_network.net_output, feed_dict={self.plan_network.net_input: rapid_trajectory.input_data})
        loss = self.sess.run( self.plan_network.loss, feed_dict={ self.plan_network.net_input: rapid_trajectory.input_data,
                                                                  self.plan_network.target_output: rapid_trajectory.output_data } )
        pos_error = tf_out_data[:,range(0,3)] - rapid_trajectory.output_data[:,range(0,3)]
        spd_error = tf_out_data[:,range(3,6)] - rapid_trajectory.output_data[:,range(3,6)]
        acc_error = tf_out_data[:,range(6,9)] - rapid_trajectory.output_data[:,range(6,9)]

        pos_error_norm = np.linalg.norm(pos_error, keepdims=1, axis=1)
        spd_error_norm = np.linalg.norm(spd_error, keepdims=1, axis=1)
        acc_error_norm = np.linalg.norm(acc_error, keepdims=1, axis=1)
        if(np.max(pos_error_norm)>pos_torlerance and
                np.mean(pos_error_norm) > pos_torlerance/5):
            print('Pos bigger than threshold: [Max, Mean, Min] = [%.3f, %.3f, %.3f] ' % (np.max(pos_error_norm), np.mean(pos_error_norm), np.min(pos_error_norm)))


        print('Loss =  ', loss)
        self.planning_summary(pos_error, spd_error, acc_error, tf_out_data)

        if (if_display):
            rapid_trajectory.plot_data("comparison")
        if(0):
            import pickle
            offline_data = dict()
            offline_data.update({ 'in': rapid_trajectory.input_data, "tr": rapid_trajectory.output_data, 'dl': tf_out_data} )
            pickle.dump(offline_data, open('comp_planning.pkl', 'wb'))
        if (if_display):
            rapid_trajectory.output_data = tf_out_data
            rapid_trajectory.plot_data_dash("comparison")
            plt.show()
        return pos_error, spd_error, acc_error, tf_out_data
