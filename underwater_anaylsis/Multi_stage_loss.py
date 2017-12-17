import matplotlib.pyplot as plt
import numpy as np
import cv2
font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 12,
        }

loss_file = '../checkpoints/Compare/underwater_pix2pix512_Res9Gmultibranch46D_selectDdcpL1a30lu5gan1_lsgan/loss_log.txt'

G_GAN = []
D_fake = []
D_real = []
G_underwater = []
D_underwater = []
stop_eopch = 65
with open(loss_file, 'r') as f:
    for line in f.readlines():
        part = line.split(' ')
        if part[0]== '(epoch:' and int(part[1][:-1])<=stop_eopch:
            G_GAN.append(float(part[9]))
            G_underwater.append(float(part[11]))
            D_real.append(float(part[13]))
            D_fake.append(float(part[15]))
            D_underwater.append(float(part[17]))

D_GAN = (np.array(D_real) + np.array(D_fake))*0.5
G_GAN_ave = G_GAN.copy()
D_fake_ave = D_GAN.copy()
G_UI_ave = G_underwater.copy()
D_UI_ave = D_underwater.copy()
for i in range(0,len(G_GAN)):
    if i < 50:
        G_GAN_ave[i] = np.mean(G_GAN[0:i + 1])
        D_fake_ave[i] = np.mean(D_fake[0:i + 1])
        G_UI_ave[i] = np.mean(G_underwater[0:i + 1])
        D_UI_ave[i] = np.mean(D_underwater[0:i + 1])
    else:
        G_GAN_ave[i] = np.mean(G_GAN[i-50:i+1])
        D_fake_ave[i] = np.mean(D_fake[i - 50:i+1])
        G_UI_ave[i] = np.mean(G_underwater[i - 50:i+1])
        D_UI_ave[i] = np.mean(D_underwater[i - 50:i+1])

lit = np.linspace(0, stop_eopch+1, len(G_GAN))

ax=plt.subplot(2,1,1)
ax.grid(True, which='both', alpha=0.2)
plt.plot(lit, G_GAN, color='crimson',alpha=0.5)
G_GAN_plt, =plt.plot(lit, G_GAN_ave, color='crimson', linewidth=2.0)
plt.xlim(0,stop_eopch)
plt.xlabel('epoch',)
plt.ylabel('loss',fontdict=font_label)
# plt.legend([G_GAN_plt], [r'$G$:$\mathcal{L}_{lscGAN}$'], loc='upper right', fontsize=12)

# ax=plt.subplot(4,1,2)
# ax.grid(True, which='both', alpha=0.2)
plt.plot(lit, D_fake,color='green',alpha=0.5)
D_GAN_plt, = plt.plot(lit, D_fake_ave, color='green',linewidth=2.0)
plt.xlim(0,stop_eopch)
plt.xlabel('epoch',fontdict=font_label)
plt.ylabel('loss',fontdict=font_label)
plt.legend([G_GAN_plt, D_GAN_plt], ['G: Ad-loss', 'D: Ad-loss'], loc='upper right',fontsize=12)

ax=plt.subplot(2,1,2)
ax.grid(True, which='both', alpha=0.2)
plt.plot(lit, G_underwater,color='blue',alpha=0.5)
G_underwater_plt, = plt.plot(lit, G_UI_ave, color='blue',linewidth=2.0)
plt.xlim(0,stop_eopch)
plt.xlabel('epoch', fontdict=font_label)
plt.ylabel('loss', fontdict=font_label)

plt.plot(lit, D_underwater, color='m', alpha=0.5)
D_underwater_plt, = plt.plot(lit, D_UI_ave, color='m',linewidth=2.0)
plt.xlim(0,stop_eopch)
plt.xlabel('epoch', fontdict=font_label)
plt.ylabel('loss', fontdict=font_label)
plt.legend([G_underwater_plt, D_underwater_plt], ['G: U-loss', 'D: U-loss'], loc='upper right',fontsize=12)

plt.show()
