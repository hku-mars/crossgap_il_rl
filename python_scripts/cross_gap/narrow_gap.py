import numpy as np
import copy
# For plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits import mplot3d

import transforms3d
import math
from numpy import linalg as LA  # For LA.norm
import sys
print(sys.platform)
if(sys.platform != "win32"):
    try:
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')  # Capactable with ros
    except:
        pass
import os

WORK_DIR = os.getenv("CROSS_GAP_WORK_DIR") # Getting path from system variable.
if WORK_DIR is None:
    WORK_DIR = "../"

sys.path.append("%s/cross_gap/"%WORK_DIR)
sys.path.append("%s/cross_gap/tools"%WORK_DIR)
sys.path.append("%s/deep_drone"%WORK_DIR)
sys.path.append("%s"%WORK_DIR)

import cv2 as cv2
import quadrocoptertrajectory as quadtraj


def draw_arrow(axes3d, pt_start, pt_direct, size=0, label='', color='b'):
    if size != 0:
        arrow_scale = size / (LA.norm(pt_direct) + 0.00000001)
    else:
        arrow_scale = 1
    axes3d.quiver(pt_start[0], pt_start[1], pt_start[2], pt_direct[0] * arrow_scale, pt_direct[1] * arrow_scale, pt_direct[2] * arrow_scale, color=color, label=label)


class Quadrotor_painter():
    def init_contour(self):
        self.motor_center = np.zeros([3, 4])
        self.motor_center[:, 0] = [self.scale, self.scale, 0];
        self.motor_center[:, 1] = [self.scale, -self.scale, 0];
        self.motor_center[:, 2] = [-self.scale, -self.scale, 0];
        self.motor_center[:, 3] = [-self.scale, self.scale, 0];

        # print(self.motor_center)
        self.axis_contour = [[], [], []]
        self.axis_contour[0] = np.zeros([2, 3])
        self.axis_contour[1] = np.zeros([2, 3])
        self.axis_contour[2] = np.zeros([2, 3])
        self.axis_contour[0][0] = self.motor_center[:, 0].T
        self.axis_contour[0][1] = self.motor_center[:, 2].T
        self.axis_contour[1][0] = self.motor_center[:, 1].T
        self.axis_contour[1][1] = self.motor_center[:, 3].T
        self.axis_contour[2][0] = [0, 0, 0]
        self.axis_contour[2][1] = [2, 0, 0]

        # print(self.axis_contour[1])

        rad2angle = 180.0 / math.pi
        step = 100
        rotor_size = 0.5 * self.scale
        theta = np.linspace(-rad2angle * 180, rad2angle * 180, 100)
        self.rotor_contour = [[], [], [], []]
        self.rotor_contour[0] = np.zeros([step, 3])
        self.rotor_contour[1] = np.zeros([step, 3])
        self.rotor_contour[2] = np.zeros([step, 3])
        self.rotor_contour[3] = np.zeros([step, 3])
        for i in range(step):
            for j in range(4):
                self.rotor_contour[j][i] = (self.motor_center[:, j] + [rotor_size * math.cos(theta[i]), rotor_size * math.sin(theta[i]), 0]).T

    def __init__(self, ax3d, scale=1):
        self.ax3d = ax3d
        self.scale = scale
        self.init_contour()
        self.R = np.eye(3, 3)
        self.center = [0, 0, 0]

    def contour_apply_RT(self, contour):
        shape = contour.shape
        plot_contour = np.zeros(shape)
        for i in range(shape[0]):
            vt = contour[i]
            vt = self.R * (np.matrix(vt).T) + np.matrix(self.center).T
            plot_contour[i, :] = (vt.T.tolist()[0])
        return plot_contour

    def plot(self):

        # print("plot ",self.R)
        for i in range(2):
            plot_contour = self.contour_apply_RT(self.axis_contour[i])
            self.ax3d.plot3D(plot_contour[:, 0], plot_contour[:, 1], plot_contour[:, 2], 'b')

        for i in range(2):
            plot_contour = self.contour_apply_RT(self.rotor_contour[i])
            self.ax3d.plot3D(plot_contour[:, 0], plot_contour[:, 1], plot_contour[:, 2], 'r')

        for i in range(2, 4):
            plot_contour = self.contour_apply_RT(self.rotor_contour[i])
            self.ax3d.plot3D(plot_contour[:, 0], plot_contour[:, 1], plot_contour[:, 2], 'b')

    def plot_plane_rot(self, center, R):
        self.R = R
        self.center = center

        self.plot()
        draw_arrow(self.ax3d, self.center, self.R[:,2], 1, '', 'g')

    def plot_plane(self, center, x, y, z):
        # print(x)
        # print(y)
        # print(z)
        R = np.eye(3, 3)
        R[0, :] = x
        R[1, :] = y
        R[2, :] = z
        self.R = copy.copy(R.T)
        self.center = center
        self.plot()


class narrow_gap():
    def np_matrix_to_float_list(self, mat):
        return mat.reshape(1, len(mat)).tolist()[0]

    def init_vertex(self):
        self.vertex_3d_vec = []
        self.vertex_3d_vec.append([0, self.rect_w / 2, self.rect_h / 2])
        self.vertex_3d_vec.append([0, -self.rect_w / 2, self.rect_h / 2])
        self.vertex_3d_vec.append([0, -self.rect_w / 2, -self.rect_h / 2])
        self.vertex_3d_vec.append([0, self.rect_w / 2, -self.rect_h / 2])

        self.vertex_3d_vec.append([0, self.rect_w / 2, self.rect_h / 2])

        for i in range(self.vertex_3d_vec.__len__()):
            pt = self.world2gap_R * np.matrix(self.vertex_3d_vec[i]).T + np.matrix(self.center).T
            self.vertex_3d_vec[i] = [float(pt[0]), float(pt[1]), float(pt[2])]
        self.vertex_3d_array = np.array(self.vertex_3d_vec)

    def init_base(self):
        # As same as the define in paper “Aggressive Quadrotor Flight through Narrow Gaps with Onboard Sensing and Computing using Active Vision”
        self.base_e = [[], [], []]
        self.base_e[0] = self.world2gap_R * np.matrix([0, 1, 0]).T  # Wider side
        self.base_e[1] = self.world2gap_R * np.matrix([1, 0, 0]).T  # Cross gap
        self.base_e[2] = np.cross(self.base_e[0].T, self.base_e[1].T).T  # e_3 = e_1 X e_2

        # print("==== base ====")
        # print(self.base_e[0], "\n-----\n", self.base_e[1], "\n-----\n", self.base_e[2], "\n-----\n")

    def set_plot_figure(self, ax):
        self.ax3d = ax
        self.quadrotor_painter = Quadrotor_painter(self.ax3d, 0.2)

    def auto_cross_path_l_d(self):
        resolution = 1000
        D_MIN = self.cross_path_d_min
        D_MAX = 1.0
        L_MIN = 0.00001
        L_MAX = 0.5

        step_l = (L_MAX - L_MIN) / resolution
        step_d = (D_MAX - D_MIN) / resolution
        print("L and D = ", step_l, step_d)
        optimal_l = L_MIN
        optimal_d = L_MAX
        min_vel = 10000000

        p_g = np.matrix(self.center).T

        for idx_l in range(0,resolution):
            temp_l = float(idx_l) * step_l + L_MIN
            # for idx_d in range(0,20):
            #     temp_d = float(idx_d) * step_d + D_MIN

            temp_d = D_MIN
            temp_t_c = math.sqrt(2 * temp_l / LA.norm(self.g_sub_vec[0]))
            temp_cross_path_p_0 = p_g - temp_l * self.base_e[0] - temp_d * self.base_e[1]
            temp_cross_path_v_0 = (temp_l / temp_t_c - 0.5 * self.norm_with_notation(self.g_sub_vec[0], self.base_e[0]) * temp_t_c) * self.base_e[0] + (
                    temp_d / temp_t_c - 0.5 * self.norm_with_notation(self.g_sub_vec[1], self.base_e[1]) * temp_t_c) * self.base_e[1]

            if(np.linalg.norm(temp_cross_path_v_0) < min_vel and temp_t_c > 0.00001):
                min_vel = np.linalg.norm(temp_cross_path_v_0)
                # print("Optimal l and d = ", temp_l, temp_d)
                # print("Initial Speed = ", np.round(temp_cross_path_v_0, 2).transpose())
                # print("Initial Speed norm = ", np.linalg.norm(temp_cross_path_v_0))
                optimal_l = temp_l
                optimal_d = temp_d

        self.cross_path_l = optimal_l
        self.cross_path_d = optimal_d

    def __init__(self, center, yaw_pitch_roll):
        self.center = center
        self.euler_ypr = yaw_pitch_roll
        self.rect_w = 2
        self.rect_h = 1
        self.world2gap_R = transforms3d.euler.euler2mat(yaw_pitch_roll[0], yaw_pitch_roll[1], yaw_pitch_roll[2], axes="szyx")
        self.init_vertex()
        self.init_base()
        self.para_cross_spd = 2

        self.g_vec = np.matrix([[0], [0], [-9.8]])
        self.g_sub_vec = [[], [], []]

        for i in range(3):
            self.g_sub_vec[i] = self.vector_project(self.g_vec, self.base_e[i])
            # print(self.base_e[i].T, "  ", g_sub_vec[i].T)

        if (sys.platform == 'win32'):
            # if(1):
            self.cross_path_d_min = 0.1
            self.auto_cross_path_l_d()
        else:
            # self.cross_path_l = 0.03
            # self.cross_path_d = 0.3
            # self.cross_path_l = 0.03
            # self.cross_path_d = 0.05
            self.cross_path_d_min = 0.1
            self.auto_cross_path_l_d()

        p_g = np.matrix(self.center).T

        self.t_c = math.sqrt(2 * self.cross_path_l / LA.norm(self.g_sub_vec[0]))
        self.cross_path_p_0 = p_g - self.cross_path_l * self.base_e[0] - self.cross_path_d * self.base_e[1]
        self.cross_path_v_0 = (self.cross_path_l / self.t_c - 0.5 * self.norm_with_notation(self.g_sub_vec[0], self.base_e[0]) * self.t_c) * self.base_e[0] + (
                self.cross_path_d / self.t_c - 0.5 * self.norm_with_notation(self.g_sub_vec[1], self.base_e[1]) * self.t_c) * self.base_e[1]

        # self.cross_ballistic_trajectory()

        print("==== crossgap summary =====")
        print("Cost time = %.3f s" % self.t_c)
        print("cross_path_l = ", self.cross_path_l)
        print("cross_path_d = ", self.cross_path_d)
        print("Initial Position = ", np.round(self.cross_path_p_0,2).transpose())
        # print("Initial Speed = ", np.round(self.cross_path_v_0).transpose())
        print("Initial Speed = ", np.round(self.cross_path_v_0,2).transpose())
        print("Initial Speed norm = ", np.linalg.norm(self.cross_path_v_0))
        print("==== crossgap summary end =====")

    def set_gap_center(self, center, yaw_pitch_roll):
        self.center = center
        self.euler_ypr = yaw_pitch_roll
        self.world2gap_R = transforms3d.euler.euler2mat(yaw_pitch_roll[0], yaw_pitch_roll[1], yaw_pitch_roll[2], axes="szyx")
        self.init_vertex()
        self.init_base()

    def draw_rotate_rectangle(self):
        self.ax3d.text(self.center[0], self.center[1], self.center[2], "Gap", color='red')
        self.ax3d.plot3D(self.vertex_3d_array[:, 0], self.vertex_3d_array[:, 1], self.vertex_3d_array[:, 2], 'gray', label="gap")

    def cos_theta_g_e(self, g, _e):
        e = _e / (LA.norm(_e) + 0.00000001)
        cos_theta = float((g.T * e) / (LA.norm(g) * LA.norm(e) + 0.00000001))
        return cos_theta

    def vector_project(self, e1, e2):  # e1 project on e2, e1 and e2 most be a vector (shape = [x,1])
        e2 = e2 / (LA.norm(e2) + 0.00000001)
        cos_theta = self.cos_theta_g_e(e1, e2)
        g_proj = LA.norm(e1) * cos_theta * e2
        return g_proj

    def norm_with_notation(self, vec, e):
        cos_theta = self.cos_theta_g_e(vec, e)
        if (cos_theta > 0):
            return LA.norm(vec)
        else:
            return -LA.norm(vec)

    def cross_ballistic_trajectory(self, if_draw=0):

        self.cross_path_v_0 = (self.cross_path_l / self.t_c - 0.5 * self.norm_with_notation(self.g_sub_vec[0], self.base_e[0]) * self.t_c) * self.base_e[0] + (
                self.cross_path_d / self.t_c - 0.5 * self.norm_with_notation(self.g_sub_vec[1], self.base_e[1]) * self.t_c) * self.base_e[1]

        if (if_draw):
            self.ax3d.text(0, 0, 0, "O", color='black')
            self.ax3d.text(2, 0, 0, "x", color='black')
            self.ax3d.text(0, 2, 0, "y", color='black')
            self.ax3d.text(0, 0, 2, "z", color='black')

            draw_arrow(self.ax3d, [0, 0, 0], [1, 0, 0], 2, label='', color='r')
            draw_arrow(self.ax3d, [0, 0, 0], [0, 1, 0], 2, label='', color='g')
            draw_arrow(self.ax3d, [0, 0, 0], [0, 0, 1], 2, label='', color='b')

            self.ax3d.scatter3D(self.cross_path_p_0[0], self.cross_path_p_0[1], self.cross_path_p_0[2], marker="^")
            t_plot = np.linspace(0, 2 * self.t_c, 1000)

            trajectory_pt_plot = []
            g_sub_vec_sub_12 = self.g_vec - self.g_sub_vec[2]
            idx = 0
            Pitch_vector = []
            Roll_vector = []
            for t in t_plot:
                pt_3d_new = self.cross_path_p_0 + self.cross_path_v_0 * t + 0.5 * g_sub_vec_sub_12 * t * t
                trajectory_pt_plot.append([float(pt_3d_new[0]), float(pt_3d_new[1]), float(pt_3d_new[2])])

                if (np.mod(idx, 200) == 0):
                    # print(pt_3d_new)
                    if (t < self.t_c):
                        plane_to_center = np.matrix(self.center).T - pt_3d_new

                        Pitch_vector = plane_to_center - self.vector_project(plane_to_center, np.matrix(-self.base_e[2]))
                        Pitch_vector = Pitch_vector / LA.norm(Pitch_vector)

                        Roll_vector = np.matrix(np.cross(Pitch_vector.T, self.base_e[2].T)).T
                        Roll_vector = Roll_vector / LA.norm(Roll_vector)

                    draw_arrow(self.ax3d, pt_3d_new.T.tolist()[0], self.base_e[2].T.tolist()[0], 0.5, '', 'r')
                    self.quadrotor_painter.plot_plane(pt_3d_new.T.tolist()[0], (Pitch_vector).T.tolist()[0], (Roll_vector).T.tolist()[0], (-self.base_e[2]).T.tolist()[0])
                    # self.quadrotor_painter.plot_plane(pt_3d_new.T.tolist()[0], (Roll_vector).T.tolist()[0], (Pitch_vector).T.tolist()[0], (self.base_e[2]).T.tolist()[0])
                idx = idx + 1

            trajectory_pt_plot = np.array(trajectory_pt_plot)
            draw_arrow(self.ax3d, self.cross_path_p_0, self.cross_path_v_0, 1, "init_speed", 'black')
            self.ax3d.plot3D(trajectory_pt_plot[:, 0], trajectory_pt_plot[:, 1], trajectory_pt_plot[:, 2], 'y--', label='cross_path')
            self.ax3d.legend()

    def approach_trajectory(self, pos_start, vel_start, acc_start, if_draw=1):
        pos0 = [float(pos_start[0]), float(pos_start[1]), float(pos_start[2])]  # position
        vel0 = [float(vel_start[0]), float(vel_start[1]), float(vel_start[2])]  # position
        acc0 = [float(acc_start[0]), float(acc_start[1]), float(acc_start[2])]  # position

        posf = self.np_matrix_to_float_list(self.cross_path_p_0)
        velf = self.np_matrix_to_float_list(self.cross_path_v_0)
        # accf = [0, 0, -9.80]  # acceleration
        # accf = [0, 0, -9.80]  # acceleration
        # accf = self.np_matrix_to_float_list(self.base_e[2]*-1.0)
        accf = self.np_matrix_to_float_list(self.g_vec - self.g_sub_vec[2])
        # accf = self.base_e[2].T.tolist()[0]  # acceleration
        gravity = [0, 0, -9.80]

        self.rapid_traj = quadtraj.RapidTrajectory(pos0, vel0, acc0, gravity)
        self.rapid_traj.set_goal_position(posf)
        self.rapid_traj.set_goal_velocity(velf)
        self.rapid_traj.set_goal_acceleration(accf)
        print('Pos0 = ', pos0)
        print('Posf = ', posf)

        print('Norm  = ', LA.norm(np.matrix(np.matrix(posf) - np.matrix(pos0))) )
        self.Tf = LA.norm(np.matrix(np.matrix(posf) - np.matrix(pos0))) / self.para_cross_spd
        self.rapid_traj.generate(self.Tf)


        self.rapid_nn_input_vector = np.zeros((17,1))
        self.rapid_pos_err = self.cross_path_p_0 - np.matrix(pos0).T
        self.rapid_pos_start = np.matrix(pos0).T
        self.rapid_spd_start = np.matrix(vel0).T
        self.rapid_acc_start = np.matrix(acc0).T
        self.rapid_spd_end = np.matrix(velf).T
        self.rapid_acc_end = np.matrix(accf).T
        self.rapid_cross_time = self.Tf

        self.rapid_nn_input_vector[1,:] = self.para_cross_spd
        self.rapid_nn_input_vector[(2, 3, 4), :] = self.rapid_pos_err
        self.rapid_nn_input_vector[(5, 6, 7), :] = self.rapid_spd_start
        self.rapid_nn_input_vector[(8, 9, 10), :] = self.rapid_acc_start
        self.rapid_nn_input_vector[(11, 12, 13), :] = self.rapid_spd_end
        self.rapid_nn_input_vector[(14, 15, 16), :] = self.rapid_acc_end
        # print(self.rapid_nn_input_vector)

        if (if_draw):

            numPlotPoints = 100
            time = np.linspace(0, self.Tf, numPlotPoints)
            position = np.zeros([numPlotPoints, 3])
            velocity = np.zeros([numPlotPoints, 3])
            acceleration = np.zeros([numPlotPoints, 3])
            thrust = np.zeros([numPlotPoints, 1])
            ratesMagn = np.zeros([numPlotPoints, 1])

            plane_acc = np.zeros([numPlotPoints, 3])
            plane_x = np.zeros([numPlotPoints, 3])
            plane_y = np.zeros([numPlotPoints, 3])
            plane_z = np.zeros([numPlotPoints, 3])

            for i in range(numPlotPoints):
                t = time[i]
                position[i, :] = self.rapid_traj.get_position(t)
                velocity[i, :] = self.rapid_traj.get_velocity(t)
                acceleration[i, :] = self.rapid_traj.get_acceleration(t)
                plane_acc[i, :] = acceleration[i, :] - gravity
                thrust[i] = self.rapid_traj.get_thrust(t)

                vec_camera2center = np.matrix(np.array(self.center) - position[i, :]).T
                Pitch_vec = (vec_camera2center - (self.vector_project(vec_camera2center, np.matrix(plane_acc[i, :]).T)))
                plane_x[i, :] = Pitch_vec.T.tolist()[0]
                plane_y[i, :] = np.cross(Pitch_vec.T, np.matrix(plane_acc[i, :])).tolist()[0]
                plane_z[i, :] = -plane_acc[i, :]

                plane_x[i, :] = plane_x[i, :] / LA.norm(plane_x[i, :])
                plane_y[i, :] = plane_y[i, :] / LA.norm(plane_y[i, :])
                plane_z[i, :] = plane_z[i, :] / LA.norm(plane_z[i, :])

                if (np.mod(i, 10) == 0 or i == numPlotPoints-1):
                    draw_arrow(self.ax3d, position[i, :], plane_acc[i, :], 1, label='', color='r')
                    # draw_arrow(self.ax3d, position[i, :], plane_x[i, :], 2, label='', color='g')
                    # draw_arrow(self.ax3d, position[i, :], plane_y[i, :], 1, label='', color='b')

                    self.quadrotor_painter.plot_plane(position[i, :], plane_x[i, :], plane_y[i, :], plane_z[i, :])
                # ratesMagn[i] = np.linalg.norm(rapid_traj.get_body_rates(t))

            self.ax3d.plot3D(position[:, 0], position[:, 1], position[:, 2], 'g--', label="approach_path")
            self.ax3d.legend()

    def get_position(self, t):
        if (t < self.Tf):
            return self.rapid_traj.get_position(t)
        else:
            g_sub_vec_sub_12 = self.g_vec - self.g_sub_vec[2]
            t = t-self.Tf
            pt_3d_new = self.cross_path_p_0 + self.cross_path_v_0 * t + 0.5 * g_sub_vec_sub_12 * t * t
            return  [float(pt_3d_new[0]), float(pt_3d_new[1]), float(pt_3d_new[2])]

    def get_velocity(self, t):
        if (t < self.Tf):
            return self.rapid_traj.get_velocity(t)
        else:
            t = t-self.Tf
            g_sub_vec_sub_12 = self.g_vec - self.g_sub_vec[2]
            # return np.matrix(self.cross_path_v_0 + self.g_vec * t).T.tolist()[0]
            return np.matrix(self.cross_path_v_0 + g_sub_vec_sub_12 * t).T.tolist()[0]

    def get_acceleration(self,t):
        if (t < self.Tf):
            return self.rapid_traj.get_acceleration(t)
        else:
            t = t-self.Tf

            g_sub_vec_sub_12 = self.g_vec - self.g_sub_vec[2]
            # return np.matrix(self.g_vec).T.tolist()[0]
            return np.matrix(g_sub_vec_sub_12).T.tolist()[0]

    def get_rotation(self, t):
        if (t< self.Tf):
            return  np.eye(3,3)
        else:

            plane_to_center = np.matrix([1,0,0]).T

            Pitch_vector = plane_to_center - self.vector_project(plane_to_center, np.matrix(-self.base_e[2]))
            Pitch_vector = Pitch_vector / LA.norm(Pitch_vector)

            Roll_vector = np.matrix(np.cross(Pitch_vector.T, self.base_e[2].T)).T
            Roll_vector = Roll_vector / LA.norm(Roll_vector)
            Rot_res = np.eye(3,3)
            Rot_res[:, 0] = np.matrix(Pitch_vector).T.tolist()[0];
            Rot_res[:, 1] = np.matrix(Roll_vector).T.tolist()[0];
            Rot_res[:, 2] = np.matrix(self.base_e[2]).T.tolist()[0];

            return Rot_res


        # Unit test
if __name__ == "__main__":
    rad2angle = 180.0 / math.pi

    fig = plt.figure()
    plt.tight_layout()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0, 15)
    ax.set_ylim(-7.5, 7.5)
    ax.set_zlim(0, 15)

    narrow_gap = narrow_gap([12, 5, 10], [0 / rad2angle, 0 / rad2angle, 30 / rad2angle]);

    narrow_gap.set_plot_figure(ax)
    narrow_gap.draw_rotate_rectangle()

    narrow_gap.cross_ballistic_trajectory();
    narrow_gap.approach_trajectory([0, 0, 0], [0, 0, 0], [0, 0, -9.8])

    plt.pause(1)

    # Define the goal raw_state:

    plt.show()
    exit()
