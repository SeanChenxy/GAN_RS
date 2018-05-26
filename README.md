<img src='imgs/GAN_RS_demo.gif' align="right" width=1000>

<br><br><br>

# GAN-Based Restoration Scheme for Underwater Vision

Our scheme adaptively restore underwater vision in real time. During training, FRS serve as the ground truth, but the output is more superior owing to our improvements.

For underwater vision, our contribution contains:
* Underwater branch in D: distinguish an image is aquatic or not
* Underwater index loss: what D shuold learn and G shuold reduce
* DCP loss: encourage the output to be similar with the ground truth in terms of dark channel
* Multi-stage loss stratage: when to ally the underwater index loss

This project is based on [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and the video result is located at [GAN-RS](https://www.youtube.com/watch?v=JKCedUhAM6M). If you use this code for your research, please cite:

Towards Qualitaty Advancement of Underwater Machine Vision with Generative Adversarial Networks,
Xingyu Chen, Junzhi Yu, Shihan Kong, Zhengxing Wu, Xi Fang, Li Wen,
arXiv preprint arXiv:1712.00736 (2017).


Image-to-Image Translation with Conditional Adversarial Networks  
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros)   
In CVPR 2017.
