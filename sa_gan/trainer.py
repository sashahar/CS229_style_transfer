import os
import time
import torch
import datetime
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from sagan_models import Generator, Discriminator, ConvolutionalGenerator, UpDownConvolutionalGenerator
from utils import *

class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss
        self.conv_G = config.conv_G

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.cuda = torch.cuda.is_available() #and cuda
        print("Using cuda:", self.cuda)

        self.build_model()

        #self.use_tensorboard = True
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()



    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):

            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            try:
                items = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                items = next(data_iter)

            X, Y = items
            #print(Y.shape)
            fake_class = torch.Tensor(np.ones(Y.shape)* np.random.randint(0, 6, size=(Y.shape[0], 1, 1, 1))).cuda()
            X, Y = X.type(torch.FloatTensor), Y.type(torch.FloatTensor)
            #X, Y = Variable(X.cuda()), Variable(Y.cuda())
            X, Y = Variable(X), Variable(Y)
            if self.cuda:
                X = X.cuda()
                Y = Y.cuda()

            #FRITS: the real_disc_in consists of the images X and the desired class
            #desired class chosen randomly, different from real class Y
            real_disc_in = torch.cat((X,Y), dim = 1)
            generator_in = torch.cat((X,fake_class), dim = 1)
            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            #real_images = tensor2var(real_images)
            #Frits TODO: why feed the real_disc_in to D?
            d_out_real,dr1,dr2 = self.D(real_disc_in)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax

            #Changed to input both image and class
            fake_images,gf1,gf2 = self.G(real_disc_in)
            fake_disc_in = torch.cat((fake_images, Y), dim = 1)
            d_out_fake,df1,df2 = self.D(fake_disc_in)

            # if self.adv_loss == 'wgan-gp':
            #     d_loss_fake = d_out_fake.mean()
            if self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()


            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # ================== Train G and gumbel ================== #
            # Create random noise
            #z = tensor2var(torch.randn(real_images.size(0), self.z_dim))

            #TODO Fritz: Do we need this?
            fake_images,_,_ = self.G(generator_in)
            fake_disc_in = torch.cat((fake_images, Y), dim = 1)
            # Compute loss with fake images
            g_out_fake,_,_ = self.D(fake_disc_in)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_loss: {:.4f}, g_loss {:.4f}"
                      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step , d_loss.item(), g_loss_fake.item(),
                             self.G.attn1.gamma.mean().item(), self.G.attn2.gamma.mean().item() ))
                    # format(elapsed, step + 1, self.total_step, (step + 1),
                    #        self.total_step , d_loss.data[0], g_loss_fake.data[0],
                    #        self.G.attn1.gamma.mean().data[0], self.G.attn2.gamma.mean().data[0] ))
                with open('log_info.txt', 'a') as f:
                    # f.write("Step {}, D Loss {}, G Loss {}\n".format(step + 1, d_loss.data[0], g_loss_fake.data[0]))
                    f.write("Step {}, D Loss {}, G Loss {}\n".format(step + 1, d_loss.item(), g_loss_fake.item()))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images,_,_= self.G(generator_in)
                result = torch.cat((X, fake_images, Y), dim = 2)
                save_image(denorm(result.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

    def build_model(self):
        self.G = None
        if self.conv_G:
            self.G = UpDownConvolutionalGenerator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim)
            if self.cuda:
                self.G = self.G.cuda()
        else:
            self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim)
            if self.cuda:
                self.G = self.G.cuda()
        self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim)
        if self.cuda:
            self.D = self.D.cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
