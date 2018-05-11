CUDA_VISIBLE_DEVICES='2,3' pythonc3 ../test_video.py \
--dataroot ../datasets/underwater/Video/2.MP4 \
--writename ../datasets/underwater/Video/2_realA.avi \
--name underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan \
--model test \
--which_model_netG 'resnet_9blocks' \
--which_direction AtoB \
--dataset_mode single \
--norm batch \
--results_dir ../results/video \
--checkpoints_dir ../checkpoints/Compare \
--which_epoch 65 \
--loadSize 512 \
--fineSize 512 \
--gpu_id '0'
#--show_video
#--writename ../datasets/underwater/Video/2_realA.mp4 \

