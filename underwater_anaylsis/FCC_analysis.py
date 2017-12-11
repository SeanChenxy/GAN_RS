import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import numpy as np
import cv2

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

dir_path = './FCC/'
file_name = ['k30.txt', 'k60.txt', 'k90.txt']
image_name = ['3.jpg', 'k30.png', 'k60.png', 'k90.png']
title = ['Low FI', 'Middle FI', 'High FI']

# file3 = open(dir_path+file_name[0], 'r')
# file6 = open(dir_path+file_name[1], 'r')
# file9 = open(dir_path+file_name[2], 'r')

k30 = np.loadtxt(dir_path+file_name[0])
k60 = np.loadtxt(dir_path+file_name[1])
k90 = np.loadtxt(dir_path+file_name[2])

k_list = [k30, k60, k90]
image_list = [dir_path+image_name[1], dir_path+image_name[2], dir_path+image_name[3]]
ori = cv2.cvtColor(cv2.imread(dir_path+image_name[1]), cv2.COLOR_BGR2RGB)
plt.figure()
gs = gridspec.GridSpec(3, 4)
ax=plt.subplot(gs[0:2, 0])
plt.imshow(ori)
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Original frame', fontdict=font_title)
ax=plt.subplot(gs[2, 0])
ax.grid(True, which='both', alpha=0.2)
num_bins = 256
o_b = ori[:,:,2].reshape(ori.shape[0]*ori.shape[1],)
o_g = ori[:,:,1].reshape(ori.shape[0]*ori.shape[1],)
o_r = ori[:,:,0].reshape(ori.shape[0]*ori.shape[1],)
plt.hist(o_b, num_bins, normed=1, facecolor='blue', alpha=0.8)
plt.hist(o_g, num_bins, normed=1, facecolor='green', alpha=0.9)
plt.hist(o_r, num_bins, normed=1, facecolor='red', alpha=0.9)
plt.xlabel('Pixel value',fontdict=font_label)
plt.ylabel('Normalized frequency', fontdict=font_label)
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
print('original: g-r: '+ str(np.mean(o_g)-np.mean(o_r)) + ' ,g/r: '+str(np.mean(o_g)/np.mean(o_r))
      + ' ,stdg: '+str(np.std(o_g))+ ' ,stdr: '+str(np.std(o_r)) )
for i, k in enumerate(k_list):
    w_b_30, w_g_30, w_r_30, n_b_30, n_g_30, n_r_30 = k
    mu = np.mean(w_b_30)  # 样本均值
    sigma = np.std(w_b_30)  # 样本标准差
    # num_bins = 256#int(max(w_b_30) - min(w_b_30)) # 区间数量(实际样本值是从100到150, 所以分成50份)
    ax = plt.subplot(gs[0, i + 1])
    plt.imshow(cv2.cvtColor(cv2.imread(image_list[i]), cv2.COLOR_BGR2RGB))
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title[i], fontdict=font_title)

    ax = plt.subplot(gs[1, i+1])
    ax.grid(True, which='both', alpha=0.2)
    plt.hist(w_b_30, num_bins, normed=1, facecolor='blue', alpha=0.8)
    plt.hist(w_g_30, num_bins, normed=1, facecolor='green', alpha=0.9)
    plt.hist(w_r_30, num_bins, normed=1, facecolor='red', alpha=0.9)
    plt.xlim(0, 50000)
    plt.xlabel('Pixel value',fontdict=font_label)
    plt.ylabel('Normalized frequency',fontdict=font_label)
    formatter = ticker.ScalarFormatter(useMathText=True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))

    ax = plt.subplot(gs[2, i+1])
    ax.grid(True, which='both', alpha=0.2)
    plt.hist(n_b_30, num_bins, normed=1, facecolor='blue', alpha=0.8)
    plt.hist(n_g_30, num_bins, normed=1, facecolor='green', alpha=0.9)
    plt.hist(n_r_30, num_bins, normed=1, facecolor='red', alpha=0.9)
    plt.xlabel('Pixel value',fontdict=font_label)
    plt.ylabel('Normalized frequency',fontdict=font_label)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    print(file_name[i] + ': g-r: ' + str(np.mean(w_g_30) - np.mean(w_r_30)) + ' ,g/r: ' + str(np.mean(w_g_30) / np.mean(w_r_30))
      + ' ,stdg: ' + str(np.std(w_g_30)) + ' ,stdr: ' + str(np.std(w_r_30)))
    print(file_name[i]+ 'norm. : g-r: ' + str(np.mean(n_g_30) - np.mean(n_r_30)) + ' ,g/r: ' + str(np.mean(n_g_30) / np.mean(n_r_30))
      + ' ,stdg: ' + str(np.std(n_g_30)) + ' ,stdr: ' + str(np.std(n_r_30)))

# plt.plot(bins, y, 'r--')
# plt.xlabel('Values')
# plt.ylabel('Probability')
# plt.title(r'$\mu={}$, $\sigma={}$'.format(round(mu, 2), round(sigma, 2)))
# plt.subplots_adjust(left=0.15)
plt.show()
# w_b_3 = list(file3.readline())
# w_b_3_np = np.array(np.long(w_b_3))
# print(w_b_3)

# file3.close()
# file6.close()
# file9.close()