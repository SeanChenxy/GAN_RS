import time
import os
from options.test_options import TestOptions
# from data.data_loader import CreateDataLoader
from models.models import create_model
import torch
import cv2
import numpy as np
# from util.visualizer import Visualizer
# from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

model = create_model(opt)
cap = cv2.VideoCapture(opt.dataroot)
if opt.writename:
    # writer = cv2.VideoCapture(opt.writename)
    fps = 50  # 视频帧率
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    size = (opt.fineSize, opt.fineSize)
    out = cv2.VideoWriter(opt.writename, fourcc, fps, size)

frame_num = 0
time_cost = 0.
# print(cap.isOpened())
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    data_resize = cv2.resize(frame, (opt.fineSize, opt.fineSize))

    data_np = cv2.normalize(data_resize, None,alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_32FC3).transpose(2,0,1)
    # cv2.imshow('test1', data_np)
    # cv2.waitKey(0)
    # data_batch = data_np.reshape(opt.batchSize, data_np.shape[2], data_np.shape[0],
    #                              data_np.shape[1] )
    data_tensor = torch.unsqueeze(torch.from_numpy(data_np),dim=0)\
        .type(torch.FloatTensor).cuda()
    data = {'A_paths': [], 'A': data_tensor}
    time1 = time.time()
    model.set_input(data)
    model.test()
    time2 = time.time()
    if(frame_num > 5):
        time_cost += time2- time1
    visuals = model.get_current_visuals()
    image_numpy = visuals['fake_B']
    out.write(image_numpy)
    if (opt.show_video):
        cv2.imshow('test', image_numpy)
        cv2.waitKey(1)
    frame_num += 1

cap.release()
if opt.writename:
    out.release()
fps = (frame_num-5)/time_cost
print('total frame: ' + str(frame_num) +'; average fps: ' + str(fps))
