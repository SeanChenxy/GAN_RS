import torch
import torch.nn as nn
import functools
import numpy as np

def define_D(input_nc, input_size, ndf, which_model_netD, n_layers_D=3, use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, 1, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, 1, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'multiout':
        netD = NLayerDiscriminator(input_nc, 3, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'multibranch':
        netD = TwoBranchNLayerDiscriminator(input_nc, input_size, 1, ndf, n_layers_b1=4, n_layers_b2=6, norm_layer=norm_layer,
                                   use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    # if use_gpu:
        # netD.cuda(device_id=gpu_ids[0])
    # init_weights(netD, init_type=init_type)
    return netD

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        pre_feature_size = 512

        self.convs = {}
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
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            self.convs['Conv'+str(n+1)] = [2, kw, padw, int(pre_feature_size)]
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

        sequence += [nn.Conv2d(ndf * nf_mult, output_nc,kernel_size=kw, stride=stw, padding=padw)]

        self.convs['Conv' + str(n_layers + 2)] = [stw, kw, padw, int(pre_feature_size)]
        pre_feature_size = (pre_feature_size + 2 * padw - kw) / stw + 1

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)


class TwoBranchNLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, input_size, output_nc=1, ndf=64, n_layers_b1=3, n_layers_b2=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(TwoBranchNLayerDiscriminator, self).__init__()
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
        print(self.convs)
        self.final_feature_size = branch_2.final_feature_size
        print(self.final_feature_size)
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
            return self.branch(x), self.branch_2(x)


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

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def receive_field(x, y, stride, padding, kernel_size):

    x_min = (x - 1) * stride + 1 - padding
    y_min = (y - 1) * stride + 1 - padding
    x_max = (x - 1) * stride - padding + kernel_size
    y_max = (y - 1) * stride - padding + kernel_size

    return x_min, y_min, x_max, y_max

def get_rf_original(x, y, convs):

    total_layers = len(convs)
    x_min, y_min, x_max, y_max = x,y,x,y
    for n_layer in range(total_layers, 0, -1):
        stride, kenel_size, padding, pre_feature_size = convs['Conv'+str(n_layer)]
        print(n_layer, stride, kenel_size, padding, pre_feature_size)

        x_min, y_min, _, _ = receive_field(x_min, y_min, stride, padding, kenel_size)
        _, _, x_max, y_max = receive_field(x_max, y_max, stride, padding, kenel_size)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(pre_feature_size, x_max), min(pre_feature_size, y_max)

    return x_min, y_min, x_max, y_max


# x_min, y_min, x_max, y_max
rf = np.zeros((30,30,4))
# print(rf[0,0].shape)
n_layers_D = 4
netD = define_D(6, 512, 64, 'multibranch', n_layers_D, False, [])
print_network(netD)
# print(netD.convs)
# print(get_rf_original(0, 0, netD.convs))


# for i in range(30):
#     for j in range(30):
#         rf[i][j] = get_rf_original(i, j, netD.convs)
#
# print(rf[0,0,0], rf[0 ,0, 1],rf[0,0,2],rf[0,0,3])
# print(rf[29,28,0], rf[29,28, 1],rf[29,28,2],rf[29,28,3])