import time
import os
from options.test_options import TestOptions
from models import create_model
import torch
import cv2
from util import util

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
# device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

model = create_model(opt)
model.setup(opt)

write_dir = os.path.join(opt.results_dir, opt.dataroot.split('/')[-1])
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

frame_num = 0
time_cost = 0
for file in os.listdir(opt.dataroot):
    image = cv2.imread(os.path.join(opt.dataroot, file))
    original_size = image.shape
    print(file, original_size)

    data_resize = cv2.resize(image, (opt.fineSize, opt.fineSize))
    data_np = cv2.normalize(data_resize, None,alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_32FC3).transpose(2,0,1)
    data_tensor = torch.unsqueeze(torch.from_numpy(data_np),dim=0).type(torch.FloatTensor)
    data = {'A_paths': [], 'A': data_tensor}
    time1 = time.time()
    model.set_input(data)
    model.test()
    time2 = time.time()
    if frame_num > 5:
        time_cost += time2- time1
    visuals = model.get_current_visuals()
    image_numpy = util.tensor2im(visuals['fake_B'])
    frame_num += 1
    cv2.imwrite(os.path.join(write_dir, file), cv2.resize(image_numpy, (original_size[1], original_size[0])))


fps = (frame_num-5)/time_cost
print('total frame: ' + str(frame_num) +'; average fps: ' + str(fps))