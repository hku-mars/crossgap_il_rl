import numpy as np
import cv2 as cv2
import airsim
import threading
import time
from termcolor import colored
from .img_tools import img_tools
from threading import Thread, Lock

class Airsim_state_capture():

    def __init__(self,client):
        self.client = client
        self.capture_mutex = Lock()

        self.img_combi = np.zeros([100, 100])
        self.state = self.client.getMultirotorState()
        self.cost_time = 0
        self.state_vector = []
        self.cost_time_vector = []

    def run_thread(self):
        self.service_thread_capture =  threading.Thread(target=self.service_capture)
        self.service_thread_capture.start()
        print(colored("Airsim capture", "blue"), colored("run as a service.", "red"))  # color print
        return
        self.service_thread_image = threading.Thread(target=self.service_capture_img)
        self.service_thread_image.start();
        print(colored("Airsim image capture", "blue"), colored("run as a service.", "red"))  # color print
        self.service_thread_state = threading.Thread(target=self.service_capture_state)
        self.service_thread_state.start();
        print(colored("Airsim raw_state capture", "blue"), colored("run as a service.", "red"))  # color print

    def service_capture_img(self):
        while(1):
            self.capture_mutex.acquire()
            self.capture_image()
            self.capture_mutex.release()

    def service_capture_state(self):
        while(1):
            self.capture_mutex.acquire()
            self.capture_state()
            self.capture_mutex.release()


    def service_capture(self):
        while(1):
            t_start = cv2.getTickCount()
            self.capture_mutex.acquire()
            self.capture_state()
            self.capture_image()
            self.capture_mutex.release()


            self.cost_time = ((cv2.getTickCount() - t_start) * 1000.0 / cv2.getTickFrequency())
            # self.cost_time_vector.append(cost_time)
            # self.state_vector.append(self.raw_state)
            # print("Query raw_state cost time = %.2f" % cost_time)
            # self.update["Get_data_cost_time": cost_time]

    def capture_state(self):
        t_start = cv2.getTickCount()
        self.state = self.client.getMultirotorState()
        # self.collision_info = self.client.simGetCollisionInfo()
        # print("Query raw_state cost time = %.2f" % ((cv2.getTickCount() - t_start) * 1000.0 / cv2.getTickFrequency()))

    def capture_image(self):
        t_start = cv2.getTickCount()
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis),  # depth visualization image
            airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True),  # depth in perspective projection
            airsim.ImageRequest("1", airsim.ImageType.Scene)])  # scene vision image in png format
        # print('Retrieved images: %d', len(responses))

        for response in responses:
            if response.pixels_as_float:
                self.img_f_raw = img_tools.process_float_img(response)
                self.img_f = img_tools.float_img_to_display(self.img_f_raw)
                # img_f = img_tools.displat_float_img( img_tools.process_float_img(response))
                # cv2.imshow("img_float", img_tools.displat_float_img(img_f))
            elif response.compress:  # png format
                self.img_png = img_tools.process_compress_img(response)
                # cv2.imshow("img_png", img_png)
                pass
            else:  # uncompressed array
                self.img_rgba = img_tools.process_rgba_img(response)
                # cv2.imshow("img_rgba", img_rgba)
        try:
            self.img_f = np.uint8(self.img_f)
            self.img_f_rgb = cv2.cvtColor(self.img_f, cv2.COLOR_GRAY2RGB)
            self.img_combi = np.concatenate((self.img_png, 255 - self.img_f_rgb), axis=0)
            # print(vis.shape)
            # cv2.imshow("image", self.img_combi)

        except Exception as e:
            print(e)
            # print(img_f_rgb.shape, img_png.shape)
            pass
        # print("Query image cost time = %.2f" % ((cv2.getTickCount() - t_start) * 1000.0 / cv2.getTickFrequency()))
        return self.img_combi
        # cv2.waitKey(1)
