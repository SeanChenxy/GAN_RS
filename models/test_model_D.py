from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch


class TestModelD(BaseModel):
    def name(self):
        return 'TestModelD'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        print(self.Tensor)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout,
                                      opt.init_type,
                                      self.gpu_ids)

        use_sigmoid = opt.no_lsgan
        self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.fineSize, opt.ndf,
                                      opt.which_model_netD, opt.n_layers_D, opt.n_layers_U, opt.norm, use_sigmoid,
                                      opt.init_type, self.gpu_ids)

        self.load_network(self.netG, 'G', opt.which_epoch)
        self.load_network(self.netD, 'D', opt.which_epoch)
        self.which_model_netD = opt.which_model_netD

        print('---------- Networks G initialized -------------')
        networks.print_network(self.netG)
        print('---------- Networks D initialized -------------')
        networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)
        self.discrim_real = self.netD.forward(torch.cat((self.real_A, self.real_B), 1))
        self.discrim_fake = self.netD.forward(torch.cat((self.real_A, self.fake_B), 1))

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_B = util.tensor2im(self.fake_B.data)
        if isinstance(self.discrim_real, tuple):
            adversarial_feature_real, underwater_feature_real = self.discrim_real
            adversarial_feature_fake, underwater_feature_fake = self.discrim_fake
            discrim_real = util.tensor2im(adversarial_feature_real.data)
            # ui_A = util.tensor2im(torch.unsqueeze(self.discrim_real.data[:,1,:,:], dim=1))
            ui_B_real = util.tensor2im(underwater_feature_real.data)
            discrim_fake = util.tensor2im(adversarial_feature_fake.data)
            ui_B_fake = util.tensor2im(underwater_feature_fake.data)
            print('ui_B_real: ')
            print(underwater_feature_real)
            print('ui_B_fake: ')
            print(underwater_feature_fake)


        else:
            discrim_real = util.tensor2im(torch.unsqueeze(self.discrim_real.data[:,0,:,:], dim=1))
            # ui_A = util.tensor2im(torch.unsqueeze(self.discrim_real.data[:,1,:,:], dim=1))
            ui_B_real = util.tensor2im(torch.unsqueeze(self.discrim_real.data[:,1,:,:], dim=1))
            discrim_fake = util.tensor2im(torch.unsqueeze(self.discrim_fake.data[:,0,:,:], dim=1))
            ui_B_fake = util.tensor2im(torch.unsqueeze(self.discrim_fake.data[:,1,:,:], dim=1))
        return OrderedDict([('real_A', real_A), ('real_B', real_B), ('fake_B', fake_B),
                            ('discrim_real', discrim_real), ('discrim_fake', discrim_fake),
                             ('ui_B_real', ui_B_real), ('ui_B_fake', ui_B_fake)])
