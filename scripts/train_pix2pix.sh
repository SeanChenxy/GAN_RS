set -ex
python ../train.py \
--dataroot '../../GAN-RS/datasets/underwater/AB640' \
--name UW_pix2pix \
--model pix2pix \
--which_model_netG unet_256 \
--which_direction AtoB \
--lambda_A 100 \
--dataset_mode aligned \
--display_id 200 \
--norm batch \
--pool_size 0
