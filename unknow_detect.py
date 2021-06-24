import sys, argparse
import numpy as np
import cv2

from detector_unknow import UnknowDetector


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--vid_path', type=str, default='')
    args = parser.parse_args()
    
    vid_path = args.vid_path
    if vid_path == '':
        print('No video file specified .')
        sys.exit(0)
    else:
        cap = cv2.VideoCapture(vid_path)
        num_frame = int(cap.get(7))
        print('num frame: ', num_frame)

        rett, start_frame = cap.read()
        '''
        初始化异物检测类，参数说明：
        sThre: 帧差结果的二值化阈值，如果错误的小框很多，可以考虑加大阈值
        min_area: 输出的框的最小面积，只会输出面积大于阈值的框
        '''
        unknown_detector = UnknowDetector(start_frame, sThre=40, min_area=1500)

        for i in range(num_frame - 1):
            ret, frame = cap.read()
            
            vis_im, unknow_objs = unknown_detector.detect_unknow(frame, i)
            '''
            vis_im: 带有异物检测框的可视化视频帧
            unknow_objs: 异物检测框, 数据结构为list，每个框以(x1, y1, x2, y3)形式存储
            '''

            if i % 10 == 0:
                print('processig frame {} ..'.format(i))
                cv2.imwrite('frames/frame_{}.jpg'.format(i), vis_im)