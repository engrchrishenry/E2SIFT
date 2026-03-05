import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .img_op import blockify, unblockify

from . import common
from . import MST
from .torch_dct import dct_2d
# sys.path.append('/storage4tb/PycharmProjects/rpg_e2vid/')
# from models_biren import common
# from models_biren import MST
# from models_biren.torch_dct import dct_2d


def blockify(image, n_blocks, block_size):
    '''image: BxCxHxW'''
    return F.unfold(image, kernel_size=block_size, stride=block_size).permute(0, 2, 1).reshape(-1, n_blocks, block_size,
                                                                                               block_size)


def unblockify(image_block, img_size, n_blocks, block_size):
    # print(image_block.permute(0,2,1).shape)
    return F.fold(image_block.reshape(-1, n_blocks, block_size ** 2).permute(0, 2, 1),
                  output_size=(img_size[0], img_size[1]), kernel_size=block_size, stride=block_size)


def rearrange_v2(img_dct):
    k = 0
    temp = torch.zeros_like(img_dct)
    for i in range(16):
        # j = 0
        temp[:, k, :, :] = img_dct[:, i, :, :]
        k = k + 1

        temp[:, k, :, :] = img_dct[:, (i + 16), :, :]
        k = k + 1

        temp[:, k, :, :] = img_dct[:, (i + 32), :, :]
        k = k + 1

        temp[:, k, :, :] = img_dct[:, (i + 48), :, :]
        k = k + 1

        temp[:, k, :, :] = img_dct[:, (i + 64), :, :]
        k = k + 1

    return temp


def dct_and_rearrange(in_pix, dct_min, dct_max):
    dct_min = dct_min.cuda()
    dct_max = dct_max.cuda()
    b, c, h_inp, w_inp = in_pix.shape
    n_blocks = (h_inp // 4) * (w_inp // 4)
    x_pix_2 = torch.zeros_like(in_pix)

    for i in range(c):
        im = in_pix[:, i, :, :].unsqueeze(1)

        im = blockify(im, n_blocks, 4)
        im = dct_2d(im)

        img_dct1 = unblockify(im, [h_inp, w_inp], n_blocks, 4)
        x_pix_2[:, i, :, :] = img_dct1.squeeze(1)

    x_pix_2 = F.pixel_unshuffle(x_pix_2, 4)
    x_pix_2 = rearrange_v2(x_pix_2)

    # x_pix_2 = (x_pix_2 - dct_min.view(1, 48, 1, 1)) / (dct_max.view(1, 48, 1, 1) - dct_min.view(1, 48, 1, 1))

    x_pix_2 = (x_pix_2 - dct_min.view(1, 80, 1, 1)) / (dct_max.view(1, 80, 1, 1) - dct_min.view(1, 80, 1, 1))

    return x_pix_2

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)

class MSTFusionBlock(nn.Module):
    def __init__(self, dim_imgfeat, dim_dctfeat, kernel_size=3, conv=default_conv):
        super(MSTFusionBlock, self).__init__()
            
        self.conv_img = nn.Sequential(conv(dim_imgfeat, dim_imgfeat, kernel_size=kernel_size),
                                      nn.ReLU(True),
                                      conv(dim_imgfeat, dim_imgfeat, kernel_size=kernel_size))
        
        self.conv_dct = nn.Sequential(conv(dim_dctfeat, dim_dctfeat, kernel_size=kernel_size),
                                      nn.ReLU(True),
                                      conv(dim_dctfeat, dim_dctfeat, kernel_size=kernel_size))
        
        self.stage_tconv = nn.ConvTranspose2d(dim_dctfeat, dim_dctfeat, kernel_size=kernel_size, stride=2, padding=(kernel_size//2))
        self.msab = MST.MSAB(dim_in=(dim_imgfeat+dim_dctfeat), dim_head=(dim_imgfeat+dim_dctfeat), dim_out=dim_imgfeat, heads=4, num_blocks=1)
    
    def forward(self, in_pix, in_dct):    
        out_pix = self.conv_img(in_pix)
        out_dct = self.conv_dct(in_dct)
        out_pix = self.msab(out_pix, self.stage_tconv(out_dct, output_size=in_pix.shape[2:]))
        #out_pix = self.msab(out_pix, out_dct)
        return out_pix+in_pix, out_dct+in_dct


#------------------------------------------------------------------------------

class TSFNet_E2SIFT(nn.Module):
    def __init__(self, dct_min_path, dct_max_path, conv=default_conv):
        super(TSFNet_E2SIFT, self).__init__()

        in_channel_img = 4 * 5
        in_channel_dct = 16 * 5
        out_channel = 4* 4

        dim_imgfeat = 96
        dim_dctfeat = 32

        kernel_size = 3
        n_basicblock = 20

        self.dct_min = torch.from_numpy(np.load(dct_min_path)).float()
        self.dct_max = torch.from_numpy(np.load(dct_max_path)).float()

        # define head module for pixel input
        self.head_pix = nn.Sequential(
            nn.Conv2d(in_channels=in_channel_img, out_channels=dim_imgfeat // 2, kernel_size=kernel_size,
                      padding=(kernel_size // 2), stride=1),
            nn.PReLU(dim_imgfeat // 2),
            nn.Conv2d(in_channels=dim_imgfeat // 2, out_channels=dim_imgfeat, kernel_size=kernel_size,
                      padding=(kernel_size // 2), stride=1)
            )

        # define head module for dct input
        self.head_dct = nn.Sequential(
            nn.Conv2d(in_channels=in_channel_dct, out_channels=dim_dctfeat // 2, kernel_size=kernel_size,
                      padding=(kernel_size // 2), stride=1),
            nn.PReLU(dim_dctfeat // 2),
            nn.Conv2d(in_channels=dim_dctfeat // 2, out_channels=dim_dctfeat, kernel_size=kernel_size,
                      padding=(kernel_size // 2), stride=1)
            )

        self.body = nn.ModuleList([MSTFusionBlock(dim_imgfeat, dim_dctfeat, kernel_size) for _ in range(n_basicblock)])

        # define tail module
        self.tail = conv(dim_imgfeat, out_channel, kernel_size)
        self.pix_shuffle = nn.PixelShuffle(2)

    def forward(self, image):

        b, c, h, w = image.shape
        # ----------------------------------------------------------------------

        img_ds = F.pixel_unshuffle(image,  2)

        img_dct = dct_and_rearrange(image, self.dct_min, self.dct_max)

        # ----------------------------------------------------------------------

        x_pix = self.head_pix(img_ds)
        x_dct = self.head_dct(img_dct)

        for i, layer in enumerate(self.body):
            if i == 0:
                res_pix, x_dct = layer(x_pix, x_dct)
            else:
                res_pix, x_dct = layer(res_pix, x_dct)

        res_pix += x_pix
        # print ('1', res_pix.shape)
        x_pix = self.tail(res_pix)
        # print ('2', res_pix.shape)
        x_pix = self.pix_shuffle(x_pix)

        return x_pix

if __name__ == '__main__':
    vox = torch.rand(1, 5, 160, 160).cuda()
    # vox_path = '/storage4tb/PycharmProjects/rpg_e2vid/output/updated/esim_reds_all/5_0.55_0.005_50_None_None/vox/002_val_00000001.npy'
    # vox = np.load(vox_path, allow_pickle=True)
    # vox = torch.from_numpy(vox).float().unsqueeze(dim=0).cuda()

    model = TSFNet_E2SIFT().cuda()
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    out = model(vox)
    print(out.shape)

