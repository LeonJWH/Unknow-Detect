import sys, time, copy
import numpy as np
import cv2

class FrameDiff():
    def __init__(self, sThre, std_img, min_area=1500):
        self.sThre = sThre
        self.std_img = std_img
        self.min_area = min_area

    def absdiff_demo(self, image_1):
        gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)  # 灰度化
        gray_image_1 = cv2.GaussianBlur(gray_image_1, (5, 5), 0)  # 高斯滤波
        gray_image_2 = cv2.cvtColor(self.std_img, cv2.COLOR_BGR2GRAY)
        gray_image_2 = cv2.GaussianBlur(gray_image_2, (5, 5), 0)
        
        d_frame = cv2.absdiff(gray_image_2, gray_image_1)
        ret, d_frame = cv2.threshold(d_frame, self.sThre, 255, cv2.THRESH_BINARY)

        return d_frame

    def lunkuo(self, img, count):

        shape = img.shape
        img_w = shape[0]
        img_h = shape[1]
        rects = []

        kernel = np.ones((12, 12), np.uint8)
        kernel1 = np.ones((12, 12), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        dilate = cv2.dilate(erosion, kernel1, iterations=1)
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if w * h > self.min_area:
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 5)
                (x1, y1, x2, y2) = (x, y, x + w, y + h)
                rects.append((x1, y1, x2, y2))

        return rects