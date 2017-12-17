import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

dir_path = '../resultsD/underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan/test_65'
path = dir_path+'/images/'

font_title = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 14,
        }
font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 10,
        }
gs = gridspec.GridSpec(2, 2)
# pre_list = ['Video4_304', 'Ancuti_1', 'Q9_922', 'Z1_2811']
pre_list = ['Video4_304']
image_name_list = []

for j, file in enumerate(pre_list):
    # if os.path.isfile(os.path.join(path, file)) == True:
    # name_split = file.split('_')
    # if name_split[2] == 'discrim.png':
    #     continue
    # class_name = name_split[2] + '_' + name_split[3]
    # if class_name != 'real_A.png':
    #     continue
    # absolute_name = name_split[0] + '_' + name_split[1]
    real_A_name = file + '_real_A.png'
    fake_B_name = file + '_fake_B.png'
    real_B_name = file + '_real_B.png'
    print(file)
    real_A = cv2.imread(path + real_A_name, 1)
    fake_B = cv2.imread(path + fake_B_name, 1)
    real_B = cv2.imread(path + real_B_name, 1)
    (w, h, _) = real_A.shape
    image_list = [real_A, real_B, fake_B]
    image_name = ['real_A', 'real_B', 'fake_B']
    for i, image in enumerate(image_list):
        plt.figure(1)
        # plt.ion()
        if i == 0:
            ax = plt.subplot(2,2,1)
            plt.imshow(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB))
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_ylabel(image_name[i], fontdict=font_label)
        # if (i == 0):
        # ax.set_title('Image', fontdict=font_title)

        image_lab = cv2.normalize(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2LAB),
                                  None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

        image_l = image_lab[:, :, 0]
        image_a = image_lab[:, :, 1]
        image_b = image_lab[:, :, 2]
        lab_bias = np.sqrt(
            np.sqrt((np.mean(image_a) - 0.5) ** 2 + (np.mean(image_b) - 0.5) ** 2) / (0.5 * np.sqrt(2)))
        lab_var = (np.max(image_a) - np.min(image_a)) * (np.max(image_b) - np.min(image_b))
        lab_light = np.mean(image_l)
        ui = np.around(lab_bias / (10 * lab_var * lab_light), decimals=3)
        print(image_name[i]+str(lab_bias)+', '+str(lab_var))
        ax = plt.subplot(2,2,i+2)
        ax.grid(True, which='both', alpha=0.2)
        image_lab_resize = cv2.resize(image_lab, (128, 128))
        plt.scatter(image_lab_resize[:,:,1]-0.5, image_lab_resize[:,:,2]-0.5)
        plt.xlim((-0.5, 0.5))
        plt.ylim((-0.5, 0.5))
        plt.xticks([-0.4,  0.4],['green', 'red'], fontsize=10)
        plt.yticks([-0.4,  0.4], ['blue', 'yellow'], fontsize=10)
        if i==0:
            ax.set_title('Original frame: U='+str(ui), fontdict=font_label)
        elif i==1:
            ax.set_title('FRS: U='+str(ui), fontdict=font_label)
        else:
            ax.set_title('GAN-RS: U='+str(ui), fontdict=font_label)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data',0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data',0))
plt.show()

