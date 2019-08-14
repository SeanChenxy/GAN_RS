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
# opt.name = 'underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan'
opt.name = 'UW_pix2pix256_unet128'
opt.model = 'test'
# opt.which_model_netG ='resnet_9blocks'
opt.which_model_netG ='unet_128'
opt.which_direction = 'AtoB'
opt.dataset_mode = 'single'
opt.norm = 'batch'
opt.results_dir = './results/video'
opt.checkpoints_dir = './checkpoints'
opt.which_epoch = 65
opt.loadSize = 512
opt.fineSize = 512

# device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
phase = 'eval'
with torch.no_grad():

    if phase == 'convert':
        model = create_model(opt)
        model.setup(opt)
        # model.eval()
        netG = model.netG.module.cpu()

        input = torch.rand(1, 3, opt.fineSize, opt.fineSize)
        script_module = torch.jit.trace(netG, input)
        output = script_module(torch.ones(1, 3, opt.fineSize, opt.fineSize))
        print(output.size())
        script_module.save("./results/netG.pt")
    elif phase == 'eval':
        model = create_model(opt)
        model.setup(opt)
        # model.eval()
        script_module = model.netG.module
        # script_module = torch.jit.load('./results/netG.pt')
        script_module = script_module.cuda()
        cap = cv2.VideoCapture('/data/UWdevkit/snippets/2.MP4')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            # print(script_module.model._modules['2'].bias.mean())
            data_resize = cv2.resize(frame, (opt.fineSize, opt.fineSize))

            data_np = cv2.normalize(data_resize, None,alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_32FC3).transpose(2,0,1)
            data_tensor = torch.unsqueeze(torch.from_numpy(data_np),dim=0).type(torch.FloatTensor).cuda()
            # data = {'A_paths': [], 'A': data_tensor}
            # model.set_input(data)
            # model.test()
            # visuals = model.get_current_visuals()
            # image_numpy = util.tensor2im(visuals['fake_B'])

            fake_B = script_module(data_tensor)
            image_numpy = util.tensor2im(fake_B)

            cv2.imshow('test', image_numpy)
            cv2.waitKey(1)

