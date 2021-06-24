import numpy as np
import cv2, sys, time, copy

import pyximport
pyximport.install()
from cython_bbox_area_percentage import bbox_area_percentage

from background_generator import BGGenerator
from frame_diff import FrameDiff
# from classifier import CLS

class UnknowDetector():
    def __init__(self, bg_std, sThre=40, min_area=1500):
        self.bg_std = bg_std
        self.sThre = sThre
        self.bg_gen = BGGenerator(min_area=min_area)
        self.frame_diff = FrameDiff(sThre, bg_std, min_area=min_area)
        # self.cls = CLS()

    def merge_and_select_boxes(self, boxes1, booxes2):
        rects_FrameDiff = np.array(boxes1)
        rects_BackgroundConstruction = np.array(booxes2)

        area_percentage = bbox_area_percentage(
            rects_FrameDiff.astype(dtype=np.float32, copy=False),
            rects_BackgroundConstruction.astype(dtype=np.float32, copy=False)
        )

        max_area_percentage = np.max(area_percentage, axis=1)
        keep_idx = np.where(max_area_percentage < 0.2)[0]
        keep_box = rects_FrameDiff[keep_idx, :]

        return keep_box


    def detect_unknow(self, frame, frame_count):
        vis_img = copy.deepcopy(frame)

        # 帧差
        d_frame = self.frame_diff.absdiff_demo(frame)
        rects_FrameDiff = self.frame_diff.lunkuo(d_frame, frame_count)

        # 背景建模        
        rects_BackgroundConstruction = self.bg_gen.bggen(frame, frame_count)

        # 背景建模和帧差的结果做IOU过滤，取帧差结果中和背景建模结果中iou小于0.2的结果
        unknow_boxes = []
        if len(rects_FrameDiff) != 0 and len(rects_BackgroundConstruction) != 0:

            keep_box = self.merge_and_select_boxes(rects_FrameDiff, rects_BackgroundConstruction)

            if keep_box.shape[0] != 0:
                for i in range(keep_box.shape[0]):
                    cv2.rectangle(
                        vis_img, 
                        (keep_box[i][0], keep_box[i][1]), 
                        (keep_box[i][2], keep_box[i][3]),
                        (0, 255, 0), 
                        thickness=3
                    )
            unknow_boxes = keep_box                 
        elif len(rects_FrameDiff) != 0 and len(rects_BackgroundConstruction) == 0:
            # keep = []
            # for box in rects_FrameDiff:
            #     tmp_im = vis_img[rects_FrameDiff[i][1]:rects_FrameDiff[i][3], rects_FrameDiff[i][0]:rects_FrameDiff[i][2]]

            for i in range(len(rects_FrameDiff)):
                cv2.rectangle(
                    vis_img, 
                    (rects_FrameDiff[i][0], rects_FrameDiff[i][1]),
                    (rects_FrameDiff[i][2], rects_FrameDiff[i][3]),
                    (0, 255, 0), 
                    thickness=3
                )
            unknow_boxes = rects_FrameDiff
        else:
            pass        
        return vis_img, unknow_boxes