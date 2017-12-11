CUDA_VISIBLE_DEVICES='0,1' pythonc3 ../test_D.py \
--dataroot ../datasets/underwater/AB640 \
--name underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan \
--model testD \
--which_model_netG resnet_9blocks \
--which_model_netD multibranch \
--init_type kaiming \
--which_direction AtoB \
--dataset_mode aligned \
--norm batch \
--results_dir ../resultsD/ \
--checkpoints_dir ../checkpoints/Compare \
--gpu_ids '1' \
--how_many 140 \
--which_epoch 65 \
--loadSize 512 \
--fineSize 512 \
--n_layers_D 4 \
--n_layers_U 6 \

#--name underwater_pix2pix512_unet256Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan \
# underwater_pix2pix512_Res9Gmultibranch45D_selectDdcpL1a30lu5gan1_lsgan
# resnet_9blocks