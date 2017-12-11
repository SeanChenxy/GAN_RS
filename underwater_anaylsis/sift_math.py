import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 12,
        }

dir_path = '../resultsD/underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan/test_65'
path = dir_path+'/images/'
ssd_path = './ssd/'
ssd_97_path = './ssd/Video3_97/'
ssd_5841_path = './ssd/Z1_5841/'

image_realA = [path+'Ancuti_1_real_A.png', path+'Ancuti_8_real_A.png']
image_realB = [path+'Ancuti_1_real_B.png', path+'Ancuti_8_real_B.png']
image_fakeB = [path+'Ancuti_1_fake_B.png', path+'Ancuti_8_fake_B.png']
image_name = [image_realA, image_realB, image_fakeB]
ssd_name = [ssd_path+'Q9_1060_real_A_ssd.png', ssd_path+'Q9_1060_real_B_ssd.png', ssd_path+'Q9_1060_fake_B_ssd.png',
            ssd_97_path+'Video3_97_real_A_ssd.png', ssd_97_path+'Video3_97_real_B_ssd.png', ssd_97_path+'Video3_97_fake_B_ssd.png',
            ssd_5841_path+'Z1_5841_real_A_ssd.png', ssd_5841_path+'Z1_5841_real_B_ssd.png', ssd_5841_path+'Z1_5841_fake_B_ssd.png']
pro = ['Original frame', 'FRS', 'GAN-RS']
label_bcd = ['(b)', '(c)', '(d)']
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
gs = gridspec.GridSpec(3, 5)

for i, name in enumerate(image_name):
    # Initiate SIFT detector
    img1 = cv2.cvtColor(cv2.imread(name[0],1), cv2.COLOR_BGR2RGB)          # queryImage
    img2 = cv2.cvtColor(cv2.imread(name[1],1), cv2.COLOR_BGR2RGB)   # trainImage
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=4)
    ax=plt.subplot(gs[i, 0:2])
    plt.imshow(img3)
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.ylabel(pro[i], fontdict=font_label)
    if (i==2):
        plt.xlabel('(a)', fontdict=font_label)

for i, name in enumerate(ssd_name):
    img = cv2.cvtColor(cv2.imread(name,1), cv2.COLOR_BGR2RGB)          # queryImage
    ax = plt.subplot(gs[i%3, i//3+2])
    plt.imshow(img)
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    if (i%3==2):
        plt.xlabel(label_bcd[i//3], fontdict=font_label)

plt.show()
