import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.special import comb
import pandas as pd

def convertToHtml(result, title):

    d = {}
    index = 0
    for t in title:
        d[t]=result[index]
        index = index+1
    df = pd.DataFrame(d)
    df = df[title]
    h = df.to_html(index=False)
    return h

# dir_path = './resultsD/underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan/test_65'
dir_path = './compare'
path =  dir_path + '/Images/'
write_file_name = dir_path+'/test.txt'
write_html_name = dir_path+'/test_html.html'
color = ('b','g','r')
font_title = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 14,
        }
font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 12,
        }



histr = []
sift = cv2.xfeatures2d.SIFT_create()
harris = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
image_name_list = []
mean_diff_list = []
average_var_list = []
average_kurtosis_list = []
laplacian_gradient_list =[]
sift_num_list = []
harris_num_list = []
canny_ratio_list = []
lab_bias_list = []
lab_var_list = []
entropy_list = []
ui_list = []
GHC_list = []
UCIQE_list = []
H_std_list = []
S_ave_list = []
ave_list = []

file_list = ['Ancuti_1', 'Ancuti_3', 'Ancuti_5', 'Im_5', 'Bali_1', 'Bali_2', 'Eustice_4', 'Q9_352',
             'Video3_97', 'Video10_5741', 'Z1_5841']
method_name_list = ['Original', 'GW', 'PB', 'CLAHE', 'DM', 'CM', 'RBLA', 'CycleGAN', 'pix2pix', 'FRS', 'GAN-RS']

gs = gridspec.GridSpec(len(method_name_list), len(file_list))
with open(write_file_name, 'w') as f:

    for cloumn, file in enumerate(file_list):
        # if os.path.isfile(os.path.join(path, file)) == True:
        #     name_split = file.split('_')
        #     if name_split[2]== 'discrim.png':
        #         continue
        #     class_name = name_split[2]+'_'+name_split[3]
        #     if class_name != 'real_A.png':
        #         continue
        #     absolute_name = name_split[0]+'_'+name_split[1]
        real_A_name = file + '_real_A.png'
        fake_B_name = file + '_fake_B.png'
        real_B_name = file + '_real_B.png'
        pb_name = file + '_pb_A.jpg'
        clahe_name = file + '_clahe_A.jpg'
        dm_name = file + '_DM_A.jpg'
        cm_name = file + '_CM_A.jpg'
        cycle_name = file + '_cycle_B.png'
        pix_name = file + '_pix_B.png'
        gw_name = file + '_gw_A.jpg'
        uir_name = file + '_UIR_A.jpg'
        dcp_name = file + '_dcp_A.jpg'
        image_name_list += [real_A_name[:-4], gw_name[:-4], pb_name[:-4], clahe_name[:-4],
                           dm_name[:-4], cm_name[:-4], uir_name[:-4], cycle_name[:-4],
                            pix_name[:-4], real_B_name[:-4], fake_B_name[:-4]]
        image_label_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
        # image_name_list.append(real_A_name[:-4])
        # image_name_list.append(fake_B_name[:-4])
        # image_name_list.append(real_B_name[:-4])
        # image_name_list.append(pb_name[:-4])
        # image_name_list.append(clahe_name[:-4])
        real_A = cv2.imread(path+real_A_name, 1)
        fake_B = cv2.imread(path+fake_B_name, 1)
        real_B = cv2.imread(path+real_B_name, 1)
        pb_A = cv2.imread(path + pb_name, 1)
        clahe_A = cv2.imread(path + clahe_name, 1)
        dm_A = cv2.imread(path + dm_name, 1)
        cm_A = cv2.imread(path + cm_name, 1)
        cycle_B = cv2.imread(path + cycle_name, 1)
        pix_B = cv2.imread(path + pix_name, 1)
        gw_A = cv2.imread(path + gw_name, 1)
        uir_A = cv2.imread(path + uir_name, 1)
        dcp_A = cv2.imread(path + dcp_name, 1)
        (w,h,_) = real_A.shape
        image_list = [real_A, gw_A, pb_A, clahe_A, dm_A, cm_A, uir_A , cycle_B, pix_B, real_B, fake_B]
        # image_name = ['real_A', 'fake_B', 'real_B', 'pb_A', 'clahe_A']

        for i, image in enumerate(image_list):
            ax=plt.subplot(gs[i, cloumn])
            # plt.figure(1)
            # plt.ion()
            # ax = plt.subplot(3, 3, 3*i+1)
            plt.imshow(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB))
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.set_xticks([])
            ax.set_yticks([])
            if cloumn==0:
                ax.set_ylabel(method_name_list[i],fontdict=font_label)
            if i==len(image_list)-1:
                ax.set_xlabel(image_label_list[cloumn],fontdict=font_title)
            # ax=plt.subplot(3, 3, 3*i+2)
            # if (i == 0):
            #     ax.set_title('Histogram',fontdict=font_title)
            ## hist
            # histr.clear()
            # for j, col in enumerate(color):
            #     histr.append(cv2.calcHist([image_list[i]], [j], None, [256], [0, 256]) / (w*h))
            #     # plt.plot(histr[-1], color=col)
            #     # plt.xlim([0, 256])
            # mean_his, var_his, kurtosis_his = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
            # for j in range(0,256):
            #     mean_his += np.array([histr[0][j] * j, histr[1][j] * j, histr[2][j] * j]).reshape([3, ])
            # # mean_his /= np.array([(w * h)]*3)
            # mean_diff_his = np.around((abs(mean_his[0]-mean_his[1])+abs(mean_his[0]-mean_his[2])+abs(mean_his[2]-mean_his[1]))/3, decimals=2)
            # for j in range(0,256):
            #     var_his += np.array([histr[0][j] * abs(j - mean_his[0]), histr[1][j] * abs(j - mean_his[1]), histr[2][j] * abs(j - mean_his[2])]).reshape([3, ])
            # b_ave = np.mean(image_list[i][:,:,0])
            # g_ave = np.mean(image_list[i][:,:,1])
            # r_ave = np.mean(image_list[i][:,:,2])
            # b_std = np.std(image_list[i][:, :, 0])
            # g_std = np.std(image_list[i][:, :, 1])
            # r_std = np.std(image_list[i][:, :, 2])
            # mean_diff = np.around((abs(b_ave-r_ave)+abs(b_ave-g_ave)+abs(g_ave-r_ave))/3, decimals=2)
            # ave_std = (b_std+g_std+r_std)/3
            # # print('      mean_channel:'+str(ave_std))
            # # var_his /= np.array([(w * h)]*3)
            # ave_var_his = np.around((var_his[0]+var_his[1]+var_his[2])/3, decimals=2)
            # for j in range(0, 256):
            #     kurtosis_his += np.array([histr[0][j] * (j - mean_his[0]) ** 4 ,
            #                           histr[1][j] * (j - mean_his[1]) ** 4,
            #                           histr[2][j] * (j - mean_his[2]) ** 4]).reshape([3, ])
            # kurtosis_his /= np.array([w*h*var_his[0]**2, w * h*var_his[1] **2, w * h*var_his[2] **2])
            # kurtosis_his -= np.array([3,3,3])
            # ave_kurtosis_his = (kurtosis_his[0] + kurtosis_his[1] + kurtosis_his[2]) / 3
            #
            # if (i==0):
            #     print('\n--------------------')
            #     print('File name: '+ file)
            #     f.write('--------------------\n')
            #     f.write('File name: ' + file +'\n')
            # print(image_name_list[i]+': ')
            # print('mean_diff of histogram: '+ str(mean_diff_his))
            # mean_diff_list.append(mean_diff_his)
            # print('average_var of histogram: '+ str(ave_var_his))
            # average_var_list.append(ave_var_his)
            # print('average_kurtosis of histogram: ' + str(ave_kurtosis_his))
            # average_kurtosis_list.append(ave_kurtosis_his)
            # f.write(image_name_list[i] + ': '+'\n')
            # f.write('mean_diff of histogram: ' + str(mean_diff_his)+'\n')
            # f.write('average_var of histogram: ' + str(ave_var_his)+'\n')
            # f.write('average_kurtosis of histogram: ' + str(ave_kurtosis_his) + '\n')

            ## laplacian and GHC
            # image_norm = cv2.normalize(image_list[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)
            laplacian = cv2.Laplacian(image_list[i], cv2.CV_32F)
            mean_laplacian = np.around(np.mean(abs(laplacian)), decimals=2)
            print('Laplacian gradient: ' + str(mean_laplacian))
            laplacian_gradient_list.append(mean_laplacian)
            f.write('Laplacian gradient: ' + str(mean_laplacian)+'\n')
            # GHC = np.around((mean_laplacian* ave_var_his)/(1+mean_diff_his), decimals=2)
            # print('GHC: ' + str(GHC))
            # GHC_list.append(GHC)
            # f.write('GHC: ' + str(GHC) + '\n')

            ## key points adn edge ##
            kp = sift.detect(image_list[i], None)
            print('SIFT number: ' + str(len(kp)))
            sift_num_list.append(len(kp))
            f.write('SIFT number: ' + str(len(kp))+'\n')
            kh = harris.detect(image_list[i], None)
            print('Harris number: ' + str(len(kh)))
            harris_num_list.append(len(kh))
            f.write('harris number: ' + str(len(kh)) + '\n')
            edges = cv2.Canny(image_list[i], 100, 200)
            print('Canny edges: ' + str(len(np.nonzero(edges)[1])/(w*h)))
            canny_ratio_list.append(len(np.nonzero(edges)[1])/(w*h))
            f.write('Canny edges: ' + str(len(np.nonzero(edges)[1]) / (w * h))+'\n')

            ## lab ##
            image_lab = cv2.normalize(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2LAB),
                                      None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)
            # image_lab = cv2.cvtColor(image_list[i],cv2.COLOR_BGR2Lab) /255.0
            # plt.figure(2)
            # plt.cla()
            # ax = plt.subplot(3, 3, 3 * i + 3)
            # plt.scatter( image_lab[:,:,1], image_lab[:,:, 2])
            # plt.xlim((0, 1))
            # plt.ylim((0, 1))
            lab_bias = np.around(np.sqrt(np.sqrt((np.mean(image_lab[:,:,1])-0.5)**2 + (np.mean(image_lab[:,:,2])-0.5)**2)/(0.5*np.sqrt(2))), decimals=2)
            lab_var = np.around((np.max(image_lab[:,:,1]) - np.min(image_lab[:,:,1])) * (np.max(image_lab[:,:,2]) - np.min(image_lab[:,:,2])), decimals=2)
            lab_light = np.mean(image_lab[:,:,0])
            ui = np.around(lab_bias / (10 * lab_var * lab_light), decimals=2)
            print("Lab_bias: "+ str(lab_bias) )
            print("Lab_var: "+ str(lab_var))
            print("UI: " + str(ui))
            f.write("Lab_bias: "+ str(lab_bias)+'\n')
            f.write("Lab_var: "+ str(lab_var)+'\n')
            f.write("UI: " + str(ui)+'\n')
            lab_bias_list.append(lab_bias)
            lab_var_list.append(lab_var)
            ui_list.append(ui)

            ## entropy ##
            tmp = []
            for p in range(256):
                tmp.append(0)
            val = 0
            k = 0
            res = 0
            image_gray = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2GRAY)
            for p in range(len(image_gray)):
                for j in range(len(image_gray[p])):
                    val = image_gray[p][j]
                    tmp[val] = float(tmp[val] + 1)
                    k = float(k + 1)
            for p in range(len(tmp)):
                tmp[p] = float(tmp[p] / k)
            for p in range(len(tmp)):
                if (tmp[p] == 0):
                    res = res
                else:
                    res = float(res - tmp[p] * (np.log(tmp[p]) / np.log(2.0)))
            entropy_list.append(np.around(res, decimals=2))
            print("Entropy: " + str(res))
            f.write("Entropy: " + str(lab_bias)+'\n')

            ## UCIQE
            # image_hsv = cv2.normalize(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2HSV_FULL),
            #                           None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)
            # h_std = np.std(image_hsv[:,:,0])
            # s_ave = np.mean(image_hsv[:,:,1])
            # print('H_std: ' + str(h_std))
            # print('S_ave: ' + str(s_ave))
            #
            # H_std_list.append(h_std)
            # S_ave_list.append(s_ave)
            #
            # UCIQE = (0.4680*h_std + 0.2576*s_ave)
            # print('UCIQE: '+str(UCIQE))
            # UCIQE_list.append(UCIQE)

ave_name_list = ['ave_real_A', 'ave_gw', 'ave_pb', 'ave_clahe',
                           'ave_DM', 'ave_CM', 'ave_UIR', 'ave_cycle',
                            'ave_pix', 'ave_real_B', 'ave_fake_B']

step = len(ave_name_list)
final = len(file_list)
for i, ave_name in enumerate(ave_name_list):
    image_name_list.append(ave_name_list[i])
    lab_bias_ave_list = lab_bias_list[i:step*(final-1)+i:step]
    # print(lab_bias_list
    # print(lab_bias_ave_list)
    # print(np.mean(lab_bias_ave_list))
    lab_bias_list.append(np.around(np.mean(lab_bias_ave_list), decimals=2))

    lab_var_ave_list = lab_var_list[i:step * (final) + i:step]
    lab_var_list.append(np.around(np.mean(lab_var_ave_list), decimals=2))

    ui_ave_list = ui_list[i:step * (final) + i:step]
    ui_list.append(np.around(np.mean(ui_ave_list), decimals=2))

    laplacian_ave_list = laplacian_gradient_list[i:step * (final) + i:step]
    laplacian_gradient_list.append(np.around(np.mean(laplacian_ave_list), decimals=2))

    entropy_ave_list = entropy_list[i:step * (final) + i:step]
    entropy_list.append(np.around(np.mean(entropy_ave_list), decimals=2))

    sift_ave_list = sift_num_list[i:step * (final) + i:step]
    sift_num_list.append(np.around(np.mean(sift_ave_list), decimals=2))

    harris_ave_list = harris_num_list[i:step * (final) + i:step]
    harris_num_list.append(np.around(np.mean(harris_ave_list), decimals=2))

    canny_ave_list = canny_ratio_list[i:step * (final) + i:step]
    canny_ratio_list.append(np.around(np.mean(canny_ave_list), decimals=2))

with open(write_html_name, 'w') as ht:
    result = [image_name_list, lab_bias_list, lab_var_list, ui_list, laplacian_gradient_list, entropy_list,
              sift_num_list, harris_num_list, canny_ratio_list
              ]
    title = ['Image name', 'Lab bias', 'Lab variation', 'UI',
             'Laplance gradient', 'Entropy',
             'SIFT points', 'Harris points', 'Edge ratio'
             ]
    ht.write(convertToHtml(result, title))

plt.show()