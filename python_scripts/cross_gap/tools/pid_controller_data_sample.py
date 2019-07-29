import numpy as np
import math, os, sys, copy
import transforms3d
import pickle
from pid_controller import Pid_controller


DATA_DIR = "%s/pid_data"%(os.getenv('Dl_data'))
print("Data save dir =  ", DATA_DIR)

try:
    os.mkdir(DATA_DIR)
except Exception as e:
    print("Make dir fail, error  =  ", e)

# In_data = [pos_err_3, spd_err_3, acc_err_3, euler_3] #12
# Out_data = [roll, pitch, yaw, thrust] #4

if __name__ == "__main__":
    print('Hi, this is pid controller data sampler')
    pid_ctrl = Pid_controller()
    pid_ctrl.maximum_attitude_angle = [85/57.3, 85/57.3, 85/57.3]
    data_size = int(1.0e6)

    save_file_name = "%s/pid_data_lim_%s.pkl"%(DATA_DIR, str(data_size))
    print(save_file_name)
    if(1):
        pos_bound = [ -0.2, 0.2 ]
        spd_bound = [ -0.3, 0.3 ]
        acc_bound = [ -10, 10 ]
    else:
        pos_bound = [ -10, 10 ]
        spd_bound = [ -5, 5 ]
        acc_bound = [ -10, 10 ]
    in_data = np.zeros(  shape=(data_size, 12) )
    out_data = np.zeros( shape=(data_size, 3)  )
    # print(in_data)
    # print(out_data)
    for loop in range(data_size):
        if (loop * 10 % data_size == 0):
            print("%d %%  " % (loop * 100 / data_size))
        euler = (np.random.uniform(low = -math.pi/2 , high = math.pi/2, size = 3 ))
        euler[2] = euler[2]*2
        # print("Euler =  ", euler*57.3 )
        q = transforms3d.euler.euler2quat(euler[0], euler[1], euler[2], axes='sxyz')

        pos_error = np.random.uniform(low = pos_bound[0] , high = pos_bound[1], size = 3 )
        spd_error = np.random.uniform(low = pos_bound[0] , high = pos_bound[1], size = 3 )
        acc_error = np.random.uniform(low = acc_bound[0] , high = acc_bound[1], size = 3 )

        pid_ctrl.update_state([0, 0, 0], [0, 0, 0], [0, 0, 0], q)
        pid_ctrl.update_target(pos_error.tolist(), spd_error.tolist(), acc_error.tolist(), target_yaw=0)
        pid_ctrl.update_control()

        input_data = np.array([pos_error, spd_error, acc_error, euler]).ravel().tolist()
        output_data  = np.array([pid_ctrl.tar_roll, pid_ctrl.tar_pitch,  pid_ctrl.tar_thrust]).ravel().tolist()

        # print("In data raw = ", np.round(input_data,2))
        # print("Out data raw= ", np.round(output_data,2))

        in_data[loop, :] = input_data
        out_data[loop, :] = output_data

        # print("In data = ", np.round(in_data[loop, :],2))
        # print("Out data = ", np.round(out_data[loop, :],2))

        # print(pos_error.ravel(), " --- ", spd_error.ravel(), " --- " , acc_error.ravel(), " --- " , euler*57.3)

    data_dict = dict({ 'in_data': in_data, 'out_data': out_data})
    print(in_data)
    print('===Out==')
    print(out_data)
    pickle.dump(data_dict, open(str(save_file_name), 'wb') )
    print('Generate data finish')