import sys, time, copy
import numpy as np
import cv2

class BGGenerator():
    def __init__(self, min_area=1500):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.min_area = min_area

    def bggen(self, frame, count):
        # frame_lwpCV = frame.copy()
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rects = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if w * h > self.min_area:
                # cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (255, 255, 0), 5)
                (x1, y1, x2, y2) = (x, y, x + w, y + h)
                rects.append((x1, y1, x2, y2))

        return rects