import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

dir_path = '../resultsD/underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan/test_65'
path = dir_path + '/images/'

image_name_list = ['Ancuti_1', 'Ancuti_8', 'Im_5', 'Video4_1258', 'Video5_4', 'Video5_235', 'Z1_2811', 'Z1_5841']

gs = gridspec.GridSpec(len(image_name_list), 7)

for j, file in enumerate(image_name_list):
    real_A_name = path + file + '_real_A.png'
    fake_B_name = path + file + '_fake_B.png'
    real_B_name = path + file + '_real_B.png'
    ad_real_name = path + file + '_discrim_real.png'
    ad_fake_name = path + file + '_discrim_fake.png'
    ui_real_name = path + file + '_ui_B_real.png'
    ui_fake_name = path + file + '_ui_B_fake.png'

    real_A = cv2.imread(real_A_name, 1)
    real_B = cv2.imread(real_B_name, 1)
    fake_B = cv2.imread(fake_B_name, 1)
    ad_real = cv2.imread(ad_real_name, 1)
    ad_fake = cv2.imread(ad_fake_name, 1)
    ui_real = cv2.imread(ui_real_name, 1)
    ui_fake = cv2.imread(ui_fake_name, 1)
    # print(real_A_name)
    # cv2.imshow('test', real_A)
    # cv2.waitKey(0)
    image_list = [real_A, real_B, fake_B, ad_real, ad_fake, ui_real, ui_fake]
    name_list = ['Origin', 'FRS', 'GAN-RS', 'FRS: ad-map', 'GAN-RS: ad-map', 'FRS: U-map', 'GAN-RS: U-map']
    for i, img in enumerate(image_list):
        # if i<3:
        ax = plt.subplot(gs[j, i])
        # else:
        #     i = i-3
        #     ax = plt.subplot(gs[2*j+1, 3*i:3*i+3])
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            plt.title(name_list[i])

    # break

plt.show()


