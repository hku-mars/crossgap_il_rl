import cv2 as cv2
import numpy as np
class img_tools:
    @staticmethod
    def process_float_img(response):
        img1d = np.asarray(response.image_data_float)
        img = img1d.reshape(response.height, response.width)

        return img

    @staticmethod
    def process_compress_img(response):
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        return  cv2.imdecode(img1d, 1);

    @staticmethod
    def float_img_to_display(_img):
        img = _img
        max_value = 1000
        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if (img[i, j] > max_value):
                    img[i, j] = max_value
        dist1 = cv2.convertScaleAbs(img)
        dist2 = cv2.normalize(dist1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return dist1
        # return dist2

    @staticmethod
    def process_rgba_img(self, response):
        # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgba = img1d.reshape(response.height, response.width, 4)  # reshape array to 4 channel image array H X W X 4
        # img_rgba = np.flipud(img_rgba)  # original image is flipped vertically
        img_rgba = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2RGB)
        return img_rgba;