import os
import shutil
fakeB_dir = '../results/underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan/test_65/images'
target_dir = '/data/UWdevkit/2017starfish/Data/train/Video_00'

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

i=0
for file in os.listdir(fakeB_dir):
    file_split = file.split('.')[0]
    file_type = file_split[-6:]
    file_name = file_split[:-7]
    print(file, file_name, file_type)
    if file_type == 'fake_B':
        i += 1
        shutil.copyfile(os.path.join(fakeB_dir, file), os.path.join(target_dir, file_name+'.jpg'))
print(i)