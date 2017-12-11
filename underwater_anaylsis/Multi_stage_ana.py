from __future__ import unicode_literals
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import numpy as np
import cv2

font_label = {'family' : 'sans-serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 10,
        }

dir_path = '../resultsD/'
net_dir = 'underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan/'
epoches = [30, 35, 45, 55, 65]#, 70, 75, 80, 85, 90, 95, 100]
show_real = True
phrase = 'test'
image_name = ['Video5_235', 'Z1_2811', 'Video4_304']
image_list = []
underwater_index = []
for epoch in epoches:
    image_dir = dir_path + net_dir + phrase + '_' + str(epoch) + '/images/'
    image_fake_name = image_dir + image_name[0] + '_fake_B.png'
    # print(image_real_name)
    if epoch == 30 and show_real:
        image_real_name = image_dir + image_name[0] + '_real_B.png'
        image_real = cv2.imread(image_real_name, 1)
        image_list.append(image_real)
        image_lab = cv2.normalize(cv2.cvtColor(image_real, cv2.COLOR_BGR2Lab),
                                  None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)
        image_l = image_lab[:, :, 0]
        image_a = image_lab[:, :, 1]
        image_b = image_lab[:, :, 2]

        lab_bias = np.sqrt(
            np.sqrt((np.mean(image_a) - 0.5) ** 2 + (np.mean(image_b) - 0.5) ** 2) / (0.5 * np.sqrt(2)))
        lab_var = (np.max(image_a) - np.min(image_a)) * (np.max(image_b) - np.min(image_b))
        lab_light = np.mean(image_l)
        ui = np.around(lab_bias / (10 * lab_var * lab_light), decimals=3)
        underwater_index.append(ui)
    image_fake = cv2.imread(image_fake_name, 1)
    image_list.append(image_fake)
    image_lab = cv2.normalize(cv2.cvtColor(image_fake, cv2.COLOR_BGR2Lab),
                   None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32FC3)
    image_l = image_lab[:,:, 0]
    image_a = image_lab[:,:, 1]
    image_b = image_lab[:, :, 2]

    lab_bias = np.sqrt(
        np.sqrt((np.mean(image_a) - 0.5) ** 2 + (np.mean(image_b) - 0.5) ** 2) / (0.5 * np.sqrt(2)))
    lab_var = (np.max(image_a) - np.min(image_a)) * (np.max(image_b) - np.min(image_b))
    lab_light = np.mean(image_l)
    ui = np.around( lab_bias / (10 * lab_var *lab_light), decimals=3)
    underwater_index.append(ui)
row = 2
col = 3
for i, itm in enumerate(image_list):
    ax = plt.subplot(row, col, i+1)
    plt.imshow(cv2.cvtColor(itm, cv2.COLOR_BGR2RGB))
    # plt.rc('text', usetex = True)
    if i == 0:
        plt.xlabel('FRS' + '; ' + r'$\Gamma$=' + str(underwater_index[i]), fontdict=font_label)
    else:
        plt.xlabel('epoch: '+ str(epoches[i-1]) +'; ' +r'$\Gamma$=' + str(underwater_index[i]), fontdict=font_label)
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
