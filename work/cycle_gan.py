import torch
from torch import nn
from torch.utils.data import DataLoader

from models.generator import build_G
from models.discriminator import build_D
from models.losses import GanLoss, CycleLoss, IdentityLoss
from utils.image_dataset import Image_Dataset
from utils.image_pool import ImagePool
from utils.logger import Logger

import itertools
import datetime
import os


class CycleGAN:
    def __init__(self):
        #设置保存路径
        self.save_root = os.path.join('checkpoints', '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')))
        if not os.path.exists(self.save_root):
            os.mkdir(self.save_root)
        #记录训练日志
        self.logger = Logger(name='cycleGAN', save_root=self.save_root)
        #————————超参数设置——————————————————
        self.epoch = 200            #迭代次数
        self.start_epoch = 0        
        self.batch_size = 2         #样本批量
        self.shuffle = False        #数据序列随机
        self.num_workers = 3        #进程数
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  #cpu/gpu调用

        self.lr = 0.0002            #学习率

        self.lambda_A = 10.         #identity_loss损失函数参数
        self.lambda_B = 10.
        self.lambda_identity = 0.5  
        #————————模型初始化————————————————
        # G_A(B) -> A 生成器_A
        self.netG_A = build_G(in_channels=3, mid_channels=64,out_channels=3, num_block=9)
        # G_B(A) -> B 生成器_B
        self.netG_B = build_G(in_channels=3, mid_channels=64,out_channels=3, num_block=9)
        # D_A(A) -> A 判别器_A
        self.netD_A = build_D(in_channels=3, mid_channels=64)
        # D_B(B) -> B 判别器_B
        self.netD_B = build_D(in_channels=3, mid_channels=64)
        #模型权重初始化
        self.init_net(init_gain = 0.02)
        self.to(self.device)

        self.label_true = torch.tensor(1.).to(self.device)
        self.label_fake = torch.tensor(0.).to(self.device)
        #————————图片缓冲池机制————————————
        self.fakeA_pool = ImagePool(100)
        self.fakeB_pool = ImagePool(100)
        #————————损失函数——————————————————
        self.gan_loss_func = GanLoss()
        self.identity_loss_func = IdentityLoss()
        self.cycle_loss_func = CycleLoss()
        #————————优化器————————————————————
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            lr=self.lr, betas=(0.5, 0.999))
        #————————学习率更新————————————————
        def scheduler_func(epoch):
            lr_l = 1.0 - max(0, epoch + self.start_epoch - self.epoch//2) / float(self.epoch//2+1)
            return lr_l
        self.schedulers_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, scheduler_func)
        self.schedulers_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, scheduler_func)

    def init_net(self, init_gain = 0.02):
        def init_model(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if hasattr(m, 'weight'):
                    nn.init.normal_(m.weight.data, 0, std=init_gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.)
        self.netG_A.apply(init_model)
        self.netG_B.apply(init_model)
        self.netD_A.apply(init_model)
        self.netD_B.apply(init_model)

    def forward(self, dataA, dataB):
        fakeA = self.netG_A(dataB)
        recB = self.netG_B(fakeA)
        fakeB = self.netG_B(dataA)
        recA = self.netG_A(fakeB)
        return fakeA, fakeB, recA, recB

    def backward_G(self, realA, realB, fakeA, fakeB, recA, recB):
        # generate loss
        loss_G_A = self.gan_loss_func(self.netD_A(fakeA), self.label_true)
        loss_G_B = self.gan_loss_func(self.netD_B(fakeB), self.label_true)
        # cycle loss
        loss_cycle_A = self.cycle_loss_func(realA, recA)
        loss_cycle_B = self.cycle_loss_func(realB, recB)
        # identity loss
        loss_idt_A = self.identity_loss_func(realA, self.netG_A(realA))
        loss_idt_B = self.identity_loss_func(realB, self.netG_B(realB))
        loss_idt_A = loss_idt_A * self.lambda_A * self.lambda_identity
        loss_idt_B = loss_idt_B * self.lambda_B * self.lambda_identity

        loss_cycle_A = loss_cycle_A * self.lambda_A
        loss_cycle_B = loss_cycle_B * self.lambda_B
        # tatol loss
        loss = loss_G_A + loss_G_B + loss_idt_A + loss_idt_B + loss_cycle_A + loss_cycle_B
        loss.backward()
        return loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_idt_B

    def backward_D(self, realA, realB, fakeA, fakeB):
        fakeA = self.fakeA_pool.query(fakeA)
        fakeB = self.fakeA_pool.query(fakeB).detach()

        fakeA = fakeA.detach()
        fakeB = fakeB.detach()

        loss_D_A_real = self.gan_loss_func(self.netD_A(realA), self.label_true)
        loss_D_A_fake = self.gan_loss_func(self.netD_A(fakeA), self.label_fake)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A.backward()

        loss_D_B_real = self.gan_loss_func(self.netD_B(realB), self.label_true)
        loss_D_B_fake = self.gan_loss_func(self.netD_B(fakeB), self.label_fake)
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B.backward()
        return loss_D_A, loss_D_B

    def save_model(self, epoch=None):
        suffix = epoch if epoch is not None else 'latest'
        torch.save(self.netG_A.cpu(), os.path.join(self.save_root, '{}_netG_A.pth'.format(suffix)))
        torch.save(self.netG_B.cpu(), os.path.join(self.save_root, '{}_netG_B.pth'.format(suffix)))
        torch.save(self.netD_A.cpu(), os.path.join(self.save_root, '{}_netD_A.pth'.format(suffix)))
        torch.save(self.netD_B.cpu(), os.path.join(self.save_root, '{}_netD_B.pth'.format(suffix)))

    def to(self, device):
        self.netG_A.to(device)
        self.netG_B.to(device)
        self.netD_A.to(device)
        self.netD_B.to(device)

    def train(self, trainA_root, trainB_root):
        #读取数据
        dataset = Image_Dataset(trainA_root, trainB_root,(256, 256), (128, 128))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        #日志记录
        self.logger.info(
            '| {:5s} | {:5s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} |'.format('EPOCH', 'ITEM', 'D_A', 'D_B','GAN_A', 'GAN_B', 'cycle_A', 'cycle_B', 'idt_A', 'idt_B', 'lr_G', 'lr_D'))
        #迭代训练
        for epoch in range(self.start_epoch+1, self.epoch + self.start_epoch+1):
            item = 0
            for dataA, dataB in dataloader:
                dataA = dataA.to(self.device)
                dataB = dataB.to(self.device)
                item += self.batch_size
                fakeA, fakeB, recA, recB = self.forward(dataA, dataB)

                self.netD_A.requires_grad_(False)
                self.netD_B.requires_grad_(False)
                self.optimizer_G.zero_grad()
                G_losses = self.backward_G(realA=dataA, realB=dataB, fakeA=fakeA, fakeB=fakeB, recA=recA, recB=recB)
                self.optimizer_G.step()

                self.netD_A.requires_grad_(True)
                self.netD_B.requires_grad_(True)
                self.optimizer_D.zero_grad()
                D_losses = self.backward_D(realA=dataA, realB=dataB, fakeA=fakeA, fakeB=fakeB)
                self.optimizer_D.step()

                if item % 100 == 0:
                    self.logger.info(
                        '| {:>5d} | {:>5d} | {:>.6f} | {:>.6f} | {:>.6f} | {:>.6f} | {:>.6f} | {:>.6f} | {:>.6f} | {:>.6f} | {:>.6f} | {:>.6f} |'.format(
                        epoch, item, D_losses[0], D_losses[1], G_losses[0], G_losses[1],
                            G_losses[2], G_losses[3], G_losses[4], G_losses[5],
                            self.optimizer_G.param_groups[0]['lr'],
                            self.optimizer_D.param_groups[0]['lr']))
            self.schedulers_G.step()
            self.schedulers_D.step()

            if epoch % 10 == 0:
                self.save_model(epoch)
                self.to(self.device)
        self.save_model()

if __name__ == '__main__':
    model = CycleGAN()
    model.train('../data/anime/trainA','../data/anime/trainB',)
