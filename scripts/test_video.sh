#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES='3' python ../test_video.py \
--dataroot /data/UWdevkit/snippets/2.MP4 \
--name underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan \
--model test \
--which_model_netG 'resnet_9blocks' \
--which_direction AtoB \
--dataset_mode single \
--norm batch \
--results_dir ../results/video \
--checkpoints_dir ../checkpoints \
--which_epoch 65 \
--loadSize 512 \
--fineSize 512 \
--gpu_id '0' \
--show_video
#--writename /data/UWdevkit/snippets/FrameByFrame/trepang/trepang_fakeB.mp4
