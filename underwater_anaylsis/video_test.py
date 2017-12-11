import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./1.jpg')
# img_lab_1 = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
image_lab = cv2.normalize(cv2.cvtColor(img, cv2.COLOR_BGR2Lab), None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)
# imageB = cv2.cvtColor(cv2.normalize(img, None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3), cv2.COLOR_BGR2Lab)
plt.scatter( image_lab[:,:,0], image_lab[:,:, 2])
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.show()
print(np.mean(image_lab[:,:,0]))
cv2.imshow('B', img)
# cv2.imshow('A', imageA)
cv2.waitKey(0)