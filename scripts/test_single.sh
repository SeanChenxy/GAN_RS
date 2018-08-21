#!/usr/bin/env bash
set -ex
CUDA_VISIBLE_DEVICES='4,5' python ../test.py \
--dataroot /home/sean/data/UWdevkit/2017/Data/train/Video_00 \
--name underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan \
--which_epoch 65 \
--fineSize 512 \
--loadSize 512 \
--checkpoints_dir ../checkpoints \
--model test \
--which_model_netG resnet_9blocks \
--which_direction AtoB \
--dataset_mode single \
--norm batch \
--results_dir ../results \
--how_many 3000
