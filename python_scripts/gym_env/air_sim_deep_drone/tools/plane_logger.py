import pickle
import numpy as np
import matplotlib.pyplot as plt
import transforms3d

class plane_logger():
    def __init__(self, name = "log.txt"):
        self.if_sub_plot = 1
        self.idx = 0
        self.data = dict()
        self.data.update({"index": []})
        self.name = name

    def update(self, key, val):
        if key in self.data.keys():
            self.data[key].append(val)
        else:
            self.data.update({key: []})
            self.data[key].append(val)
        self.data["index"].append(self.idx)
        self.idx = self.idx + 1

    def save(self):
        file = open(self.name, 'wb')
        pickle.dump(self.data, file)
        file.close()

    def load(self, name):
        file = open(name, "rb")
        self.data = pickle.load(file)
        file.close()

    def plot_data(self):
        try:
            time_stamp_vec = np.array(self.data["time_stamp"])

            # ------------------input---------------
            plt.figure("Altitude")
            current_rot_vec = np.array(self.data["current_rot"])
            target_rot_vec = np.array(self.data["target_rot"])
            shape = current_rot_vec.shape
            current_euler_vec = np.zeros((shape[0], 3))
            target_euler_vec = np.zeros((shape[0], 3))
            for k in range(shape[0]):
                rot_mat = np.matrix(current_rot_vec[k])
                r, p, y = transforms3d.euler.mat2euler(rot_mat, axes="sxyz")
                current_euler_vec[k, 0] = r*57.3
                current_euler_vec[k, 1] = p*57.3
                current_euler_vec[k, 2] = y*57.3

                rot_mat = np.matrix(target_rot_vec[k])
                r, p, y = transforms3d.euler.mat2euler(rot_mat, axes="sxyz")
                target_euler_vec[k, 0] = r*57.3
                target_euler_vec[k, 1] = p*57.3
                target_euler_vec[k, 2] = y*57.3

            plt.plot(time_stamp_vec, current_euler_vec[:, 0], "r", label='current_roll')  # ,marker ="+")
            plt.plot(time_stamp_vec, current_euler_vec[:, 1], "k", label='current_pitch')  # ,marker ="^")
            plt.plot(time_stamp_vec, current_euler_vec[:, 2], "b", label='current_yaw')  # ,marker ="v")

            plt.plot(time_stamp_vec, target_euler_vec[:, 0], "r--", label='target_roll')  # ,marker ="+")
            plt.plot(time_stamp_vec, target_euler_vec[:, 1], "k--", label='target_pitch')  # ,marker ="^")
            plt.plot(time_stamp_vec, target_euler_vec[:, 2], "b--", label='target_yaw')  # ,marker ="v")

            plt.grid()
            plt.legend()
            plt.pause(0.1)

            # ------------------Pos---------------
            if (self.if_sub_plot == 0):
                plt.figure("pos")
            else:
                plt.figure("log")
                plt.subplot(3, 1, 1)
                plt.title("pos")
            current_pos = np.array(self.data["current_pos"])[:, :]
            plt.plot(time_stamp_vec, current_pos[:, 0], "r", label='current_pos_x')  # ,marker ="+")
            plt.plot(time_stamp_vec, current_pos[:, 1], "k", label='current_pos_y')  # ,marker ="^")
            plt.plot(time_stamp_vec, current_pos[:, 2], "b", label='current_pos_z')  # ,marker ="v")

            target_pos = np.array(self.data["target_pos"])[:, :]
            plt.plot(time_stamp_vec, target_pos[:, 0], "r--", label='target_pos_x')  # ,marker ="+")
            plt.plot(time_stamp_vec, target_pos[:, 1], "k--", label='target_pos_y')  # ,marker ="^")
            plt.plot(time_stamp_vec, target_pos[:, 2], "b--", label='target_pos_z')  # ,marker ="v")

            plt.grid()
            plt.legend()
            plt.pause(0.1)
            # ------------------Spd---------------
            if (self.if_sub_plot == 0):
                plt.figure("spd")
            else:
                plt.subplot(3, 1, 2)
                plt.title("spd")

            current_spd = np.array(self.data["current_spd"])[:, :]
            plt.plot(time_stamp_vec, current_spd[:, 0], "r", label='current_spd_x')  # ,marker ="+")
            plt.plot(time_stamp_vec, current_spd[:, 1], "k", label='current_spd_y')  # ,marker ="^")
            plt.plot(time_stamp_vec, current_spd[:, 2], "b", label='current_spd_z')  # ,marker ="v")

            target_spd = np.array(self.data["target_spd"])[:, :]
            plt.plot(time_stamp_vec, target_spd[:, 0], "r--", label='target_spd_x')  # ,marker ="+")
            plt.plot(time_stamp_vec, target_spd[:, 1], "k--", label='target_spd_y')  # ,marker ="^")
            plt.plot(time_stamp_vec, target_spd[:, 2], "b--", label='target_spd_z')  # ,marker ="v")

            plt.grid()
            plt.pause(0.1)
            plt.legend()

            # ------------------acc---------------
            if (self.if_sub_plot == 0):
                plt.figure("acc")
            else:
                plt.subplot(3, 1, 3)
                plt.title("acc")

            current_acc = np.array(self.data["current_acc"])[:, :]
            plt.plot(time_stamp_vec, current_acc[:, 0], "r", label='current_acc_x')  # ,marker ="+")
            plt.plot(time_stamp_vec, current_acc[:, 1], "k", label='current_acc_y')  # ,marker ="^")
            plt.plot(time_stamp_vec, current_acc[:, 2], "b", label='current_acc_z')  # ,marker ="v")

            target_acc = np.array(self.data["target_acc"])[:, :]
            plt.plot(time_stamp_vec, target_acc[:, 0], "r--", label='target_acc_x')  # ,marker ="+")
            plt.plot(time_stamp_vec, target_acc[:, 1], "k--", label='target_acc_y')  # ,marker ="^")
            plt.plot(time_stamp_vec, target_acc[:, 2], "b--", label='target_acc_z')  # ,marker ="v")

            plt.grid()
            plt.pause(0.1)
            plt.legend()

            # ------------------Throttle---------------
            plt.figure("Throttle")
            throttle_vec = np.array(self.data["input_throttle"])
            plt.plot(throttle_vec, "k", label='input_throttle')  # , marker="o")
            plt.pause(0.1)
            plt.legend()

            # ------------------R P Y---------------
            plt.figure("roll_pitch_yaw")
            r_vec = np.array(self.data["input_roll"])
            p_vec = np.array(self.data["input_pitch"])
            y_vec = np.array(self.data["input_yaw"])
            plt.plot(r_vec, "r", label='input_roll')  # , marker="o")
            plt.plot(p_vec, "k", label='input_pitch')  # , marker="o")
            plt.plot(y_vec, "b", label='input_yaw')  # , marker="o")
            plt.pause(0.1)
            plt.legend()


            # ------------------Time perframe---------------
            plt.figure("ctrl_time")

            shape = time_stamp_vec.shape
            time_per_frame = np.zeros(shape[0]-1)
            for k in range(0, shape[0]):
                time_per_frame[k-1] = (time_stamp_vec[k] - time_stamp_vec[k-1])*1000.0

            time_per_frame[0]= time_stamp_vec[1] # init cost a lot of time?
            plt.plot(time_per_frame, "k", label='input_throttle')  # , marker="o")

            plt.pause(0.1)
            plt.legend()

            pass
        except Exception as e:
            print(e)


if __name__ == "__main__":
    print("hello")
    plane_logger = plane_logger("hello")
    plane_logger.load("../plane_log.txt")
    plane_logger.plot_data()
    plt.show()
    # print("Load:")
    # print(plane_logger.data)

    print("finish")
