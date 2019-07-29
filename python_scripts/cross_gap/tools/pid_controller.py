import numpy as np
import math, os, sys, copy
import transforms3d

bag_name = '_2019-01-29-19-51-29.bag'
bag_name = '_2019-01-30-16-26-51.bag'
# bag_name = '_2019-01-30-16-39-55.bag'
bag_name = '_2019-01-30-17-38-45.bag'
pkl_file_name = '%s/real_flight/%s.pkl'%( os.getenv('DL_data'), bag_name )

print(pkl_file_name)

class Pid_controller:
    # Cordinate FLU(forward, left, up)
    # TODO: add integration of velocity
    def load_default_parameter(self):
        Kp = [6.5, 6.5, 8.5]
        Kv = [2.25, 2.25, 4.50]
        Ka = [0.9, 0.9, 0.9]

        self.set_parameter(Kp, Kv, Ka, g = 9.8 ,mass = 0.90)
        # small test
        self.update_target([0, 0, 0.1], [0, 0, 0], [0, 0, 0], target_yaw = 0)
        self.update_state([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0, 0])
        self.update_control()

    def __init__(self):
        self.Kp = np.eye(3,3)
        self.Kv = np.eye(3,3)
        self.Ka = np.eye(3,3)
        self.K = np.eye(3,3)
        self.mass = 1.0
        self.gravity = np.array([0, 0, 9.8])

        self.pos_err = np.array([0, 0, 0])
        self.vel_err = np.array([0, 0, 0])
        self.acc_err = np.array([0, 0, 0])
        self.current_attitude = np.eye(3,3)

        self.maximum_attitude_angle = [75.0/57.3, 75.0/57.3, 75.0/57.3]

        self.tar_yaw = 0
        self.maximum_thrust = 60.0
        self.input_limitation_pos = 0.3
        self.input_limitation_vel = 2.0
        self.Ctrl_limit_f_norm = 2000
        self.load_default_parameter()

    def set_Kp(self, _v):
        self.Kp = np.array([[_v[0], 0, 0],
                            [0, _v[1], 0],
                            [0, 0, _v[2]]])

    def set_Kv(self, _v):
        self.Kv = np.array([[_v[0], 0, 0],
                            [0, _v[1], 0],
                            [0, 0, _v[2]]])

    def set_Ka(self, _v):
        self.Ka = np.array([[_v[0], 0, 0],
                            [0, _v[1], 0],
                            [0, 0, _v[2]]])

    def set_gravity(self, _v):
        self.gravity = np.array([0, 0, _v])

    def set_full_thrust(self, _thrust):
        self.maximum_thrust = _thrust

    def set_mass(self, mass):
        self.mass = mass

    def set_maximum_angle(self, angle_rad_x, angle_rad_y=0, angle_rad_z=0 ):
        if(angle_rad_y == 0):
            angle_rad_y = angle_rad_x
        if (angle_rad_z == 0):
            angle_rad_z = angle_rad_x

        self.maximum_attitude_angle = [angle_rad_x, angle_rad_y, angle_rad_z]
        print("Pid controller set maximum angle limit = ", np.round(np.array(self.maximum_attitude_angle)*57.3),2)

    def set_target_yaw(self, angle_rad):
        self.tar_yaw = angle_rad


    def update_target(self, _pos, _vel, _acc, target_yaw = 0):
        self.tar_pos = np.array([_pos[0], _pos[1], _pos[2]])
        self.tar_vel = np.array([_vel[0], _vel[1], _vel[2]])
        self.tar_acc = np.array([_acc[0], _acc[1], _acc[2]])
        self.tar_yaw = target_yaw
        # print('Tar: ', self.tar_pos.ravel(), ' -- ',  self.tar_vel.ravel(), ' -- ',  self.tar_acc.ravel() )

    def update_state(self, _pos, _vel, _acc , q):
        self.cur_pos = np.array([_pos[0], _pos[1], _pos[2]])
        self.cur_vel = np.array([_vel[0], _vel[1], _vel[2]])
        self.cur_acc = np.array([_acc[0], _acc[1], _acc[2]])

        self.cur_q = [q[0], q[1], q[2], q[3]]
        self.current_rpy = transforms3d.euler.quat2euler(self.cur_q, axes='sxyz')
        self.current_attitude = transforms3d.euler.quat2mat(self.cur_q)
        # print('Cur: ', self.cur_pos.ravel(), ' -- ',  self.cur_vel.ravel(), ' -- ',  self.cur_acc.ravel() )

    def update_control(self):
        self.pos_err = self.tar_pos - self.cur_pos
        self.vel_err = self.tar_vel - self.cur_vel
        self.acc_err = self.tar_acc

        if(np.linalg.norm(self.pos_err ) > self.input_limitation_pos ):
            self.pos_err = self.pos_err*self.input_limitation_pos / np.linalg.norm(self.pos_err)

        if(np.linalg.norm(self.vel_err ) > self.input_limitation_vel ):
            self.vel_err = self.vel_err*self.input_limitation_vel / np.linalg.norm(self.vel_err)


        self.pos_item = np.matmul(self.Kp, self.pos_err)
        self.vel_item = np.matmul(self.Kv, (self.vel_err + self.pos_item))
        self.acc_item = np.matmul(self.Ka, self.acc_err)

        self.force_desire = (self.vel_item + self.gravity) * self.mass + self.acc_item * self.mass

        param_gra = self.gravity[2]

        if (np.linalg.norm(self.force_desire) > self.Ctrl_limit_f_norm):
            self.force_desire = self.force_desire * self.Ctrl_limit_f_norm / np.linalg.norm(self.force_desire)

        if (self.force_desire[2] < 0.05 * self.mass * 9.8):
            self.force_desire = self.force_desire / self.force_desire[2] * (0.05 * self.mass * param_gra);
            self.force_desire[2] = (0.05 * self.mass * param_gra);

        # Limit pitch
        if (math.fabs( self.force_desire[0] / self.force_desire[2]  ) > math.tan(self.maximum_attitude_angle[1]) ):
            self.force_desire[0] = self.force_desire[0] / math.fabs(self.force_desire[0]) * self.force_desire[2] * math.tan(self.maximum_attitude_angle[1])

        # Limit roll
        if (math.fabs( self.force_desire[1] / self.force_desire[2]  ) > math.tan(self.maximum_attitude_angle[0]) ):
            self.force_desire[1] = self.force_desire[1] / math.fabs(self.force_desire[1]) * self.force_desire[2] * math.tan(self.maximum_attitude_angle[0])

        z_b_curr = self.current_attitude[:,2]

        self.tar_thrust = self.force_desire.dot(z_b_curr) / np.linalg.norm(z_b_curr)
        # self.tar_thrust = np.linalg.norm(self.force_desire) *100/  self.maximum_thrust
        # self.tar_thrust = self.force_desire[2] / self.maximum_thrust

        # print('*-----*')
        # print(self.current_attitude)
        # print(z_b_curr, np.linalg.norm(z_b_curr),  self.force_desire)
        # print((self.force_desire[2] / self.maximum_thrust - self.force_desire.dot(z_b_curr) / self.maximum_thrust)*100)
        # print((np.linalg.norm(self.force_desire) / self.maximum_thrust - self.force_desire.dot(z_b_curr) / self.maximum_thrust)*100)
        # self.tar_thrust
        # = 0
        wRc = transforms3d.euler.axangle2mat(axis=[0,0,1], angle=self.current_rpy[2]*1, is_normalized = True )

        force_in_c = np.matmul(wRc, self.force_desire)

        fx = force_in_c[0]
        fy = force_in_c[1]
        fz = force_in_c[2]
        z_b_des = self.force_desire / np.linalg.norm( self.force_desire)

        y_c_des = np.array([-math.sin(self.tar_yaw), math.cos(self.tar_yaw), 0.0])
        x_b_des = np.cross(y_c_des, z_b_des) / np.linalg.norm(np.cross(y_c_des, z_b_des))
        y_b_des = np.cross(z_b_des, x_b_des)

        self.tar_r_mat = np.array([x_b_des.T, y_b_des.T, z_b_des.T])

        # print(force_in_c)
        self.tar_roll = math.atan2(-fy, fz)
        self.tar_pitch = math.atan2(fx, fz)

        if(abs(self.tar_roll) > self.maximum_attitude_angle[0]):
            self.tar_roll =  self.maximum_attitude_angle[0]*self.tar_roll/abs(self.tar_roll)
        if(abs(self.tar_pitch) > self.maximum_attitude_angle[1]):
            self.tar_pitch =  self.maximum_attitude_angle[1]*self.tar_pitch/abs(self.tar_pitch)


    def set_parameter(self, Kp, Kv, Ka,  g = 9.8 ,mass = 0.90):
        self.set_Kp(Kp)
        self.set_Kv(Kv)
        self.set_Ka(Ka)
        self.set_gravity(g)
        self.set_mass(mass)
        print("==== Set parameter ====")
        print("Gravity = " , self.gravity.ravel())
        print("Mass = " , self.mass)

        print("==== Set Kp ====")
        print(self.Kp)
        print("==== Set Kv ====")
        print(self.Kv)
        print("==== Set Ka ====")
        print(self.Ka)

def draw_curve(bag_dict):
    import matplotlib.pyplot as plt
    if (1):
        plt.figure("Traj_track")
        plt.subplot(3, 1, 1)
        plt.title("Position")
        topic = '/tar_state'
        show_bias = 6
        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 1 + show_bias], '-r', label='target_x')
        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 2 + show_bias], '-g', label='target_y')
        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 3 + show_bias], '-b', label='target_z')

        topic = "/cur_state"
        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 1 + show_bias], '--r', label='current_x')
        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 2 + show_bias], '--g', label='current_y')
        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 3 + show_bias], '--b', label='current_z')

        plt.legend()
        plt.grid('on')

        plt.subplot(3, 1, 2)
        topic = '/n1ctrl/ctrl_dbg/a'
        plt.title('ctrl_desire_forces')
        data = bag_dict[topic]

        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 1], 'r-', label='desire_a_x')
        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 2], 'b-', label='desire_a_y')
        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 3], 'k-', label='desire_a_z')
        plt.plot(bag_dict[topic][:, 0], np.linalg.norm( bag_dict[topic][:, (range(1,4))] , axis=1),'y-', label= 'desire_a_norm')
        plt.legend()
        plt.grid('on')
        plt.pause(0.1)

        topic = 'simulate'
        plt.title('ctrl_desire_forces')

        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 1], 'r--', label='py_desire_a_x')
        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 2], 'b--', label='py_desire_a_y')
        plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 3], 'k--', label='py_desire_a_z')

        plt.plot(bag_dict[topic][:, 0], np.linalg.norm( bag_dict[topic][:, (range(1,4))] , axis=1),'y--', label= 'desire_a_norm')

        plt.legend()
        plt.grid('on')

        plt.subplot(3, 1, 3)
        desire_topic = '/n1ctrl/ctrl_dbg/att_des'

        plt.plot(bag_dict[desire_topic][:, 0], bag_dict[desire_topic][:, 1] * 57.3, 'r-', label='desire_roll')
        plt.plot(bag_dict[desire_topic][:, 0], bag_dict[desire_topic][:, 2] * 57.3, 'b-', label='desire_pitch')
        plt.plot(bag_dict[desire_topic][:, 0], bag_dict[desire_topic][:, 3] * 57.3, 'k-', label='desire_yaw')
        topic = '/djiros/ctrl'
        # plt.plot(bag_dict[topic][:, 0], bag_dict[topic][:, 3] , 'g-', label='desire_thrust')

        desire_topic = '/ctrl_sim'
        plt.plot(bag_dict[desire_topic][:, 0], bag_dict[desire_topic][:, 1] * 57.3, 'r--', label='py_desire_roll')
        plt.plot(bag_dict[desire_topic][:, 0], bag_dict[desire_topic][:, 2] * 57.3, 'b--', label='py_desire_pitch')
        plt.plot(bag_dict[desire_topic][:, 0], bag_dict[desire_topic][:, 3] * 57.3, 'k--', label='py_desire_yaw')

        # plt.plot(bag_dict[desire_topic][:, 0], bag_dict[desire_topic][:, 4] , 'g--', label='py_desire_thrust')


        desire_topic = '/cur_rpy'
        plt.plot(bag_dict[desire_topic][:, 0], bag_dict[desire_topic][:, 1] * 57.3, 'r-.', label='cur_roll')
        plt.plot(bag_dict[desire_topic][:, 0], bag_dict[desire_topic][:, 2] * 57.3, 'b-.', label='cur_pitch')
        plt.plot(bag_dict[desire_topic][:, 0], bag_dict[desire_topic][:, 3] * 57.3, 'k-.', label='cur_yaw')


        plt.legend()
        plt.grid('on')

        plt.pause(0)


def offline_compare(pid_ctrl):
    import pickle
    odom_topic = '/uart_odom/out_odom'
    name_list = [odom_topic, '/position_cmd', '/djiros/ctrl', '/djiros/imu',
                 '/n1ctrl/ctrl_dbg/att_des', '/n1ctrl/ctrl_dbg/att_real',
                 '/n1ctrl/ctrl_dbg/p', '/n1ctrl/ctrl_dbg/v', '/n1ctrl/ctrl_dbg/a',
                 '/n1ctrl/ctrl_dbg/omega_p', '/n1ctrl/ctrl_dbg/omega_d', '/n1ctrl/ctrl_dbg/cor_force',
                 '/cross_gap/gap_pose']

    cmd_topic= '/position_cmd'
    file = open(pkl_file_name, "rb")
    bag_dict = pickle.load(file)
    file.close()


    forces = copy.deepcopy(bag_dict[cmd_topic])
    cur_state = copy.deepcopy(bag_dict[cmd_topic])
    cur_rpy = copy.deepcopy(bag_dict[cmd_topic])
    tar_state = copy.deepcopy(bag_dict[cmd_topic])
    ctrl_out = copy.deepcopy(bag_dict[cmd_topic])

    train_data_in = copy.deepcopy(bag_dict[cmd_topic])
    train_data_out = copy.deepcopy(bag_dict[cmd_topic])

    #
    # forces[:,0] = bag_dict[cmd_topic][:,0]
    # cur_state[:,0] = bag_dict[cmd_topic][:,0]
    # cur_rpy[:,0] = bag_dict[cmd_topic][:,0]
    # tar_state[:,0] = bag_dict[cmd_topic][:,0]
    # ctrl_out[:,0] = bag_dict[cmd_topic][:,0]
    last_avail_time_index  =0

    for cmd_idx in range(0,bag_dict[cmd_topic].shape[0]):
        time_stamp = bag_dict[cmd_topic][cmd_idx, 0]
        # print(time_stamp)

        time_index = np.nonzero(np.round(bag_dict[odom_topic][:, 0],2) == np.round(time_stamp, 2))
        if (len(time_index[0]) != 0):
            last_avail_time_index = time_index[0][0]
            # print(time_stamp)
            cur_pos = [bag_dict[odom_topic][time_index[0][0], 1], bag_dict[odom_topic][time_index[0][-1], 2], bag_dict[odom_topic][time_index[0][0], 3]]
            cur_vel = [bag_dict[odom_topic][time_index[0][0], 4], bag_dict[odom_topic][time_index[0][-1], 5], bag_dict[odom_topic][time_index[0][0], 6]]
            cur_acc = [0,0,0]
            cur_q = [bag_dict[odom_topic][time_index[0][0], 7], bag_dict[odom_topic][time_index[0][-1], 8],
                     bag_dict[odom_topic][time_index[0][0], 9], bag_dict[odom_topic][time_index[0][-1], 10] ]


            tar_pos = [bag_dict[cmd_topic][cmd_idx, 1], bag_dict[cmd_topic][cmd_idx, 2], bag_dict[cmd_topic][cmd_idx, 3]]
            tar_vel = [bag_dict[cmd_topic][cmd_idx, 4], bag_dict[cmd_topic][cmd_idx, 5], bag_dict[cmd_topic][cmd_idx, 6]]
            tar_acc = [bag_dict[cmd_topic][cmd_idx, 7], bag_dict[cmd_topic][cmd_idx, 8], bag_dict[cmd_topic][cmd_idx, 9]]

            pid_ctrl.update_target(tar_pos, tar_vel, tar_acc)
            pid_ctrl.update_state(cur_pos, cur_vel, cur_acc, cur_q)
            pid_ctrl.update_control()
        else:
            pass
            # print('could not find')

        cur_state[cmd_idx, range(1, 7)] = bag_dict[odom_topic][last_avail_time_index, range(1, 7)]      # pos, spd
        cur_state[cmd_idx, range(7, 10)] = bag_dict[odom_topic][last_avail_time_index, range(7, 10)]*0  # acc = 0
        cur_state[cmd_idx, range(10, 14)] = bag_dict[odom_topic][last_avail_time_index, range(7, 11)]   # q

        tar_state[cmd_idx, range(1, 10)] = bag_dict[cmd_topic][cmd_idx, range(1, 10)]

        forces[cmd_idx, 1] = pid_ctrl.force_desire[0]
        forces[cmd_idx, 2] = pid_ctrl.force_desire[1]
        forces[cmd_idx, 3] = pid_ctrl.force_desire[2]

        ctrl_out[cmd_idx, 1] = pid_ctrl.tar_roll
        ctrl_out[cmd_idx, 2] = pid_ctrl.tar_pitch
        ctrl_out[cmd_idx, 3] = pid_ctrl.tar_yaw
        ctrl_out[cmd_idx, 4] = pid_ctrl.tar_thrust

        cur_rpy[cmd_idx, 1] = pid_ctrl.current_rpy[0]
        cur_rpy[cmd_idx, 2] = pid_ctrl.current_rpy[1]
        cur_rpy[cmd_idx, 3] = pid_ctrl.current_rpy[2]

        # Saving trainning data
        train_data_in[cmd_idx,range(0,9)] = tar_state[cmd_idx, range(1, 10)]  - cur_state[cmd_idx, range(1, 10)]
        euler_angle = transforms3d.euler.quat2euler( cur_state[cmd_idx, range(10, 14)], axes='sxyz' )
        train_data_in[cmd_idx, range(9, 12)] = euler_angle
        train_data_out[cmd_idx, 0] = pid_ctrl.current_rpy[0]
        train_data_out[cmd_idx, 1] = pid_ctrl.current_rpy[1]
        train_data_out[cmd_idx, 2] = pid_ctrl.tar_thrust

    bag_for_train = dict({'in_data': train_data_in, 'out_data': train_data_out })
    pickle.dump(bag_for_train, open('%s.pkl'%pkl_file_name, 'wb'))

    bag_dict.update({"simulate": forces})
    bag_dict.update({"/cur_state": cur_state})
    bag_dict.update({"/cur_rpy": cur_rpy})
    bag_dict.update({"/tar_state": tar_state})
    bag_dict.update({"/ctrl_sim": ctrl_out})
    draw_curve(bag_dict)

if __name__ == "__main__":
    print("Hello, this is pid_controller test")

    pid_ctrl = Pid_controller()

    offline_compare(pid_ctrl)