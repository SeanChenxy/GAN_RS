import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 12,
        }

# dir_path = '../resultsD/underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan/test_65'
path = './pic/compare/Images/'
ssd_path = './pic/ssd/Video3_97/'
color_cls = {3:'m', 1:'b', 2:'r'}
det_file = ssd_path+'res97.txt'
det_res = {'real_A':list(),'gw_A':list(), 'pb_A':list(), 'clahe_A':list(), 'DM_A':list(),
           'CM_A': list(), 'UIR_A': list(), 'pix_B': list(), 'real_B': list(), 'fake_B': list(),}
for line in open(det_file, 'r'):
    line_split = line.split(' ')
    name = line_split[0].split('.')[-2].split('_')[-2] + '_' + line_split[0].split('.')[-2].split('_')[-1]
    det_res[name].append(line_split[1:])
    print(name)

print(det_res)
image_realA = [path+'Ancuti_1_real_A.png', path+'Ancuti_8_real_A.png']
image_realB = [path+'Ancuti_1_real_B.png', path+'Ancuti_8_real_B.png']
image_fakeB = [path+'Ancuti_1_fake_B.png', path+'Ancuti_8_fake_B.png']
image_gw = [path+'Ancuti_1_gw_A.jpg', path+'Ancuti_8_gw_A.jpg']
image_pb = [path+'Ancuti_1_pb_A.jpg', path+'Ancuti_8_pb_A.jpg']
image_clahe = [path+'Ancuti_1_clahe_A.jpg', path+'Ancuti_8_clahe_A.jpg']
image_dm = [path+'Ancuti_1_DM_A.jpg', path+'Ancuti_8_DM_A.jpg']
image_cm = [path+'Ancuti_1_CM_A.jpg', path+'Ancuti_8_CM_A.jpg']
image_uir = [path+'Ancuti_1_UIR_A.jpg', path+'Ancuti_8_UIR_A.jpg']
image_pix = [path+'Ancuti_1_pix_B.png', path+'Ancuti_8_pix_B.png']


image_name = [image_realA, image_gw, image_pb, image_clahe, image_dm, image_cm, image_uir, image_pix, image_realB, image_fakeB]
ssd_name = [ssd_path+'Video3_97_real_A_ssd.png', ssd_path+'Video3_97_gw_A_ssd.png', ssd_path+'Video3_97_pb_A_ssd.png',
            ssd_path+'Video3_97_clahe_A_ssd.png', ssd_path+'Video3_97_DM_A_ssd.png', ssd_path+'Video3_97_CM_A_ssd.png',
            ssd_path+'Video3_97_UIR_A_ssd.png', ssd_path+'Video3_97_pix_B_ssd.png', ssd_path+'Video3_97_real_B_ssd.png', ssd_path+'Video3_97_fake_B_ssd.png']
pro = ['Original frame', 'GW', 'PB', 'CLAHE', 'DM', 'CM', 'RBLA', 'pix2pix', 'FRS', 'GAN-RS']
# label_bcd = ['(b)', '(c)', '(d)']
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
gs = gridspec.GridSpec(len(image_name), 3)

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
    # if (i==2):
    #     plt.xlabel('(a)', fontdict=font_label)

for i, (key, dets) in enumerate(det_res.items()):
    # print(key,dets )
    type = 'jpg' if key in ['UIR_A', 'gw_A', 'CM_A', 'DM_A', 'clahe_A', 'pb_A'] else 'png'

    name = ssd_path+'Video3_97_%s.%s' % (key, type)
    img = cv2.cvtColor(cv2.imread(name,1), cv2.COLOR_BGR2RGB)          # queryImage
    ax = plt.subplot(gs[i, 2])
    plt.imshow(img)
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    for object in dets:
        cls, score, x_min, y_min, x_max, y_max = object
        cls, x_min, y_min, x_max, y_max = int(cls), int(x_min), int(y_min), int(x_max), int(y_max)
        color = color_cls[cls]
        rect = patches.Rectangle((x_min, y_min), x_max - x_min,
                                 y_max - y_min, linewidth=1.0, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

plt.show()
