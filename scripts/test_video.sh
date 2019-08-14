#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES='3' python ../test_video.py \
--dataroot /data/UWdevkit/snippets/2.MP4 \
--name UW_pix2pix320_unet128 \
--model test \
--which_model_netG 'unet_128' \
--which_direction AtoB \
--dataset_mode single \
--norm batch \
--results_dir ../results/video \
--checkpoints_dir ../checkpoints \
--which_epoch 5 \
--loadSize 384 \
--fineSize 384 \
--gpu_id '0' \
--show_video
#--writename /data/UWdevkit/snippets/FrameByFrame/trepang/trepang_fakeB.mp4
# --name underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan \
