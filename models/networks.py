import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import cv2
import numpy as np

###############################################################################
# Helper Functions
###############################################################################


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
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'resnet_3blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_custom':
        netG = Unet()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


def define_D(input_nc, ndf, which_model_netD, n_layers_D=4,
             n_layers_U=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'multibranch':
        netD = MultiBranchDiscriminator(input_nc, ndf, n_layers_b1=n_layers_D, n_layers_b2=n_layers_U, norm_layer=norm_layer,
                                   use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class U_loss(nn.Module):
    def __init__(self, D, fineSize, use_lsgan=True, device=torch.device('cpu')):
        super(U_loss, self).__init__()
        self.device = device
        self.D = D
        self.conv_size, self.critic = self.get_conv_size(fineSize)
        self.final_size = self.conv_size.pop()
        self.conv_size.reverse()
        self.critic.reverse()
        self.Umap0 = torch.zeros((self.final_size, self.final_size), requires_grad=False).to(self.device)
        self.rf_map = np.zeros((self.final_size,self.final_size, 4)).astype(int)
        for i in range(self.final_size):  #x
            for j in range(self.final_size): #y
                self.rf_map[i, j] = self.get_rf_original(i, j)
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_conv_size(self, fineSize):
        critic = []
        for layer in list(self.D.trunk) + list(self.D.critic_branch):
            if isinstance(layer, nn.Conv2d):
                critic.append(layer)

        conv_size = [fineSize, ]
        in_size = conv_size[0]
        for conv_layer in critic:
            out_size = np.floor((in_size + 2*conv_layer.padding[0] - conv_layer.kernel_size[0])/conv_layer.stride[0] + 1)
            conv_size.append(int(out_size))
            in_size = out_size
        return conv_size, critic

    def receive_field(self, x, y, stride, padding, kernel_size):

        x_min = (x - 1) * stride + 1 - padding
        y_min = (y - 1) * stride + 1 - padding
        x_max = (x - 1) * stride - padding + kernel_size
        y_max = (y - 1) * stride - padding + kernel_size

        return x_min, y_min, x_max, y_max

    def get_rf_original(self, x, y):

        x_min, y_min, x_max, y_max = x, y, x, y
        for conv_layer, pre_feature_size in zip(self.critic, self.conv_size):
            x_min, y_min, _, _ = self.receive_field(x_min, y_min, conv_layer.stride[0], conv_layer.padding[0], conv_layer.kernel_size[0])
            _, _, x_max, y_max = self.receive_field(x_max, y_max, conv_layer.stride[0], conv_layer.padding[0], conv_layer.kernel_size[0])
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(pre_feature_size, x_max), min(pre_feature_size, y_max)

        return int(x_min), int(y_min), int(x_max), int(y_max)

    def get_Umap(self, input):
        underwater_index_batchmap = []
        for instance in input:
            # AorB = np.random.rand()
            underwater_index_map = np.zeros((1, self.final_size, self.final_size))
            image = cv2.normalize(instance.detach().cpu().float().numpy().transpose(1, 2, 0),
                                       None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
            # image = image_cat[: ,:, 3:] if AorB>0.5 else image_cat[: ,:, :3]
            image_lab = cv2.normalize(cv2.cvtColor(image, cv2.COLOR_RGB2Lab), None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

            for i in range(self.final_size): #y
                for j in range(self.final_size) : #x

                    image_sub_l = image_lab[self.rf_map[j, i, 1]:self.rf_map[j, i, 3], self.rf_map[j, i, 0]:self.rf_map[j, i, 2], 0]
                    image_sub_a = image_lab[self.rf_map[j, i, 1]:self.rf_map[j, i, 3], self.rf_map[j, i, 0]:self.rf_map[j, i, 2], 1]
                    image_sub_b = image_lab[self.rf_map[j, i, 1]:self.rf_map[j, i, 3], self.rf_map[j, i, 0]:self.rf_map[j, i, 2], 2]
                    lab_bias = np.sqrt(np.sqrt((np.mean(image_sub_a) - 0.5) ** 2 + (np.mean(image_sub_b) - 0.5) ** 2) / (0.5 * np.sqrt(2)))
                    lab_var = (np.max(image_sub_a) - np.min(image_sub_a)) * (np.max(image_sub_b) - np.min(image_sub_b))
                    lab_light = np.mean(image_sub_l)
                    underwater_index_map[0, j, i] = lab_bias / (10*lab_var*lab_light)


            underwater_index_batchmap.append(underwater_index_map)
        with torch.no_grad():
            Umap = torch.from_numpy(np.array(underwater_index_batchmap)).type(torch.FloatTensor).to(self.device)
        return Umap

    def __call__(self, image, pred_critic, model='G'):
        Umap = self.Umap0.expand_as(pred_critic) if model == 'G' else self.get_Umap(image)
        return self.loss(pred_critic, Umap)

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
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
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

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
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


# Defines the multi-branch discriminator with the specified arguments.
class MultiBranchDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers_b1=3, n_layers_b2=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(MultiBranchDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        self.trunk = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True) )
        ## ad_branch
        nf_mult = 1
        nf_mult_prev = 1
        ad_branch = []
        for n in range(1, n_layers_b1):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            ad_branch += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers_b1, 8)
        ad_branch += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        ad_branch += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            ad_branch += [nn.Sigmoid()]

        self.ad_branch = nn.Sequential(*ad_branch)

        ## critic_branch
        nf_mult = 1
        nf_mult_prev = 1
        critic_branch = []
        for n in range(1, n_layers_b2):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            critic_branch += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers_b1, 8)
        critic_branch += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        critic_branch += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            critic_branch += [nn.Sigmoid()]

        self.critic_branch = nn.Sequential(*critic_branch)

    def forward(self, input):
        x = self.trunk(input)
        return self.ad_branch(x), self.critic_branch(x)

## Unet costom

class conv_bn_relu(nn.Module):
    def __init__(self,in_channel, out_channel, stride = 1,has_relu = True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,3 ,stride = stride,padding=1,bias=True)

        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None
    def forward(self, x):
        x = self.conv(x)

        if self.relu:
            x = self.relu(x)
        return x


class BlockIn(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(BlockIn, self).__init__()
        self.conv1 = conv_bn_relu(in_channel,out_channel)
        self.conv2 = conv_bn_relu(out_channel,out_channel)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Projector(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(Projector, self).__init__()
        self.conv1 = conv_bn_relu(in_channel,out_channel, has_relu= False)
    def forward(self, x):
        x = self.conv1(x)
        return x

class Blockdown(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(Blockdown, self).__init__()
        self.conv1 = conv_bn_relu(in_channel,out_channel,stride=2)
        self.conv2 = conv_bn_relu(out_channel,out_channel)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class BlockUp(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BlockUp, self).__init__()
        self.conv1 = conv_bn_relu(in_channel,out_channel)
        self.conv_adjust = conv_bn_relu(out_channel*2,out_channel)
    def forward(self, feature_small, feature_big):
        feature_small = self.conv1(feature_small)
        f_resize = F.interpolate(feature_small,scale_factor=2,mode='bilinear',align_corners=True)
        f_cat = torch.cat([ f_resize, feature_big],1)
        f_adjust = self.conv_adjust(f_cat)
        return f_adjust


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.unet_in = BlockIn(3,32)
        self.unet_d1 = Blockdown(32,64)
        self.unet_d2 = Blockdown(64,128)
        self.unet_d3 = Blockdown(128,256)
        self.unet_d4 = Blockdown(256,512)
        self.unet_u0 = BlockUp(512,256)
        self.unet_u1 = BlockUp(256,128)
        self.unet_u2 = BlockUp(128,64)
        self.unet_u3 = BlockUp(64,32)
        self.unet_u4 = Projector(32,3)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        f_d_64 = self.unet_in(x)
        f_d_32 = self.unet_d1(f_d_64)
        f_d_16 = self.unet_d2(f_d_32)
        f_d_8  = self.unet_d3(f_d_16)
        f_d_4  = self.unet_d4(f_d_8)
        f_u_8 = self.unet_u0(f_d_4, f_d_8)
        f_u_16 = self.unet_u1(f_u_8, f_d_16)
        f_u_32 = self.unet_u2(f_u_16,f_d_32)
        f_u_64 = self.unet_u3(f_u_32,f_d_64)
        p = self.Tanh(self.unet_u4(f_u_64))

        return p

