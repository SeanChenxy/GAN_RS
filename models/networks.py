import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import cv2
import matplotlib.pyplot as plt
import util.util as util
###############################################################################
# Functions
###############################################################################



def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.step_list, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(device_id=gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, input_size, ndf, which_model_netD, n_layers_D=3, n_layers_U=4,
             norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, input_size, 1, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, input_size, 1, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'multiout':
        netD = NLayerDiscriminator(input_nc, input_size, 2, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'multibranch':
        netD = MultiBranchDiscriminator(input_nc, input_size, 1, ndf, n_layers_b1=n_layers_D, n_layers_b2=n_layers_U, norm_layer=norm_layer,
                                   use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, convD={}, final_feature_size=0, fine_size=512, batch_size=1):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.rf_map = np.zeros((final_feature_size,final_feature_size,4)).astype(int)
        self.convsD = convD
        self.final_feature_size = final_feature_size
        self.fine_size = fine_size
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

        for i in range(self.final_feature_size):  #x
            for j in range(self.final_feature_size): #y
                self.rf_map[i, j] = self.get_rf_original(i, j)
        print(self.rf_map[4, 3])

    def receive_field(self, x, y, stride, padding, kernel_size):

        x_min = (x - 1) * stride + 1 - padding
        y_min = (y - 1) * stride + 1 - padding
        x_max = (x - 1) * stride - padding + kernel_size
        y_max = (y - 1) * stride - padding + kernel_size

        return x_min, y_min, x_max, y_max

    def get_rf_original(self, x, y):

        total_layers = len(self.convsD)
        x_min, y_min, x_max, y_max = x, y, x, y
        for n_layer in range(total_layers, 0, -1):
            stride, kenel_size, padding, pre_feature_size = self.convsD['Conv' + str(n_layer)]

            x_min, y_min, _, _ = self.receive_field(x_min, y_min, stride, padding, kenel_size)
            _, _, x_max, y_max = self.receive_field(x_max, y_max, stride, padding, kenel_size)
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(pre_feature_size, x_max), min(pre_feature_size, y_max)

        return int(x_min), int(y_min), int(x_max), int(y_max)

    def get_histogram_mean_diff(self, input):
        histr = []
        for j in range(0, 3):
            histr.append( (cv2.calcHist([input], [j], None, [256], [0, 256])) )
        mean_his = np.array([0.0, 0.0, 0.0])
        for j in range(0, 256):
            mean_his += np.array([histr[0][j] * j, histr[1][j] * j, histr[2][j] * j]).reshape([3, ])
        mean_his /= np.array([self.fine_size ** 2] * 3)
        # print(abs(mean_his[0] - mean_his[1]) + abs(mean_his[0] - mean_his[2]) + abs(mean_his[2] - mean_his[1]))
        return (abs(mean_his[0] - mean_his[1]) + abs(mean_his[0] - mean_his[2]) + abs(mean_his[2] - mean_his[1]))

    def get_underwater_index_map(self, input):
        underwater_index_batchmap = []
        for instance in input:
            # AorB = np.random.rand()
            underwater_index_map = np.zeros((1, self.final_feature_size,self.final_feature_size))
            image = cv2.normalize(instance.cpu().data.float().numpy().transpose(1, 2, 0),
                                       None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
            # image = image_cat[: ,:, 3:] if AorB>0.5 else image_cat[: ,:, :3]
            image_lab = cv2.normalize(cv2.cvtColor(image, cv2.COLOR_RGB2Lab), None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

            for i in range(self.final_feature_size): #y
                for j in range(self.final_feature_size) : #x

                    image_sub_l = image_lab[self.rf_map[j, i, 1]:self.rf_map[j, i, 3], self.rf_map[j, i, 0]:self.rf_map[j, i, 2], 0]
                    image_sub_a = image_lab[self.rf_map[j, i, 1]:self.rf_map[j, i, 3], self.rf_map[j, i, 0]:self.rf_map[j, i, 2], 1]
                    image_sub_b = image_lab[self.rf_map[j, i, 1]:self.rf_map[j, i, 3], self.rf_map[j, i, 0]:self.rf_map[j, i, 2], 2]
                    lab_bias = np.sqrt(np.sqrt((np.mean(image_sub_a) - 0.5) ** 2 + (np.mean(image_sub_b) - 0.5) ** 2) / (0.5 * np.sqrt(2)))
                    lab_var = (np.max(image_sub_a) - np.min(image_sub_a)) * (np.max(image_sub_b) - np.min(image_sub_b))
                    lab_light = np.mean(image_sub_l)
                    underwater_index_map[0, j, i] = lab_bias / (10*lab_var*lab_light)


            underwater_index_batchmap.append(underwater_index_map)

        return Variable(torch.from_numpy(np.array(underwater_index_batchmap)),
                                                     requires_grad=False).type(torch.FloatTensor).cuda()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def get_noisy_target_tensor(self, input, target_is_real, model):
        target_tensor = None
        if target_is_real:
            real_tensor = self.Tensor(input.size()).fill_(self.real_label)+(0.5*torch.rand(input.size())-0.3).cuda()
            self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)+(0.3*torch.rand(input.size())).cuda()
            self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input_ori, input_pre, target_is_real, model='D', noise=False, lambda_U=10, lambda_GAN=1):

        if isinstance(input_pre, tuple) or input_pre.size()[1]>1: # multiout or multibranch
            if isinstance(input_pre, tuple):
                adversarial_feature, underwater_feature = input_pre
            else:
                adversarial_feature = input_pre[:, 0, :, :].unsqueeze(1)
                underwater_feature = input_pre[:, 1, :, :].unsqueeze(1)
            underwater_index_map = Variable(self.Tensor(underwater_feature.size()).fill_(0.), requires_grad=False) if model == 'G' else self.get_underwater_index_map(input_ori)
            adversarial_target_tensor = self.get_noisy_target_tensor(adversarial_feature, target_is_real, model) if noise else self.get_target_tensor(adversarial_feature, target_is_real)
            return self.loss(adversarial_feature, adversarial_target_tensor)*lambda_GAN, self.loss(underwater_feature, underwater_index_map)*lambda_U
        else:
            target_tensor = self.get_noisy_target_tensor(input_pre, target_is_real, model) if noise else self.get_target_tensor(input_pre, target_is_real)
            return self.loss(input_pre, target_tensor)

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, input_size, output_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        pre_feature_size = input_size
        self.convs = {}
        self.final_feature_size = pre_feature_size
        stw = 2
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stw, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        self.convs['Conv1'] = [stw, kw, padw, int(pre_feature_size)]
        pre_feature_size = (pre_feature_size + 2 * padw - kw) / stw + 1

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=stw, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            self.convs['Conv' + str(n + 1)] = [stw, kw, padw, int(pre_feature_size)]
            pre_feature_size = (pre_feature_size + 2 * padw - kw) / stw + 1

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        stw = 1
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=stw, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.convs['Conv' + str(n_layers + 1)] = [stw, kw, padw, int(pre_feature_size)]
        pre_feature_size = (pre_feature_size + 2 * padw - kw) / stw + 1

        sequence += [nn.Conv2d(ndf * nf_mult, output_nc,kernel_size=kw, stride=1, padding=padw)]
        self.convs['Conv' + str(n_layers + 2)] = [stw, kw, padw, int(pre_feature_size)]
        self.final_feature_size = int((pre_feature_size + 2 * padw - kw) / stw + 1)

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the Multi branch PatchGAN discriminator with the specified arguments.
class MultiBranchDiscriminator(nn.Module):
    def __init__(self, input_nc, input_size, output_nc=1, ndf=64, n_layers_b1=3, n_layers_b2=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(MultiBranchDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        pre_feature_size = input_size
        self.convs = {}
        self.final_feature_size = pre_feature_size
        stw = 2
        trunk = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stw, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        self.convs['Conv1'] = [stw, kw, padw, int(pre_feature_size)]
        pre_feature_size = (pre_feature_size + 2 * padw - kw) / stw + 1
        nf_mult = 1
        branch_1 = Branch(nf_mult, output_nc, ndf=ndf, n_layers=n_layers_b1, norm_layer=norm_layer,
                          use_sigmoid=use_sigmoid, use_bias=use_bias, pre_feature_size=pre_feature_size,
                          gpu_ids=gpu_ids)
        branch_2 = Branch(nf_mult, output_nc, ndf=ndf, n_layers=n_layers_b2, norm_layer=norm_layer,
                          use_sigmoid=use_sigmoid, use_bias=use_bias, pre_feature_size=pre_feature_size,
                          gpu_ids=gpu_ids)
        for key, item in branch_2.convs.items():
            assert(key not in self.convs)
            self.convs[key] = item
        # print(self.convs)
        self.final_feature_size = branch_2.final_feature_size
        # print(self.final_feature_size)
        self.trunk = nn.Sequential(*trunk)
        self.branch_1 = nn.Sequential(branch_1)
        self.branch_2 = nn.Sequential(branch_2)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            x = nn.parallel.data_parallel(self.trunk, input, self.gpu_ids)
            return nn.parallel.data_parallel(self.branch_1, x, self.gpu_ids), \
                   nn.parallel.data_parallel(self.branch_2, x, self.gpu_ids)
        else:
            x = self.trunk(input)
            return self.branch_1(x), self.branch_2(x)


class Branch(nn.Module):
    def __init__(self, pre_layer, output_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 use_bias=False, pre_feature_size=256, gpu_ids=[]):
        super(Branch, self).__init__()
        self.gpu_ids = gpu_ids
        self.branch = self.n_layer_branch(pre_layer, output_nc, ndf=ndf, n_layers=n_layers,
                                          norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                          pre_feature_size=pre_feature_size, use_bias=use_bias)

    def n_layer_branch(self, pre_layer, output_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                       pre_feature_size=256, use_bias=False):
        branch = []
        kw = 4
        padw = 1
        stw = 2
        self.convs = {}
        pre_layer = pre_layer
        nf_mult = min(2 ** (pre_layer-1), 8)
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            branch += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=stw, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            self.convs['Conv' + str(n + pre_layer)] = [stw, kw, padw, int(pre_feature_size)]
            pre_feature_size = (pre_feature_size + 2 * padw - kw) / stw + 1

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        stw = 1
        branch += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=stw, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.convs['Conv' + str(n_layers + 1)] = [stw, kw, padw, int(pre_feature_size)]
        pre_feature_size = (pre_feature_size + 2 * padw - kw) / stw + 1

        branch += [nn.Conv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)]
        self.convs['Conv' + str(n_layers + 2)] = [stw, kw, padw, int(pre_feature_size)]
        self.final_feature_size = int((pre_feature_size + 2 * padw - kw) / stw + 1)

        if use_sigmoid:
            branch += [nn.Sigmoid()]
        return nn.Sequential(*branch)

    def forward(self, input):
        return self.branch(input)