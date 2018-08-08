set -ex
CUDA_VISIBLE_DEVICES='4,5' python test.py \
--dataroot ./datasets/facades \
--name facades_pix2pix \
--model pix2pix \
--which_model_netG unet_256 \
--which_direction BtoA \
--dataset_mode aligned \
--norm batch
