from common import ResBlock, ResBlockAttn, default_conv, Upsampler
from MST_fusion import MSAB, MSAB_crossAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct_2d

def blockify(image, n_blocks, block_size):
    '''image: BxCxHxW'''
    return F.unfold(image, kernel_size=block_size, stride=block_size).permute(0,2,1).reshape(-1, n_blocks, block_size, block_size)


def unblockify(image_block, img_size, n_blocks, block_size):
    #print(image_block.permute(0,2,1).shape)
    return F.fold(image_block.reshape(-1, n_blocks, block_size**2).permute(0, 2, 1), output_size=(img_size[0], img_size[1]), kernel_size=block_size, stride=block_size)

def rearrange_v2(img_dct):
    k = 0
    temp = torch.zeros_like(img_dct)
    for i in range(16):
        j = 0
        temp[:,k,:,:]    = img_dct[:,i,:,:]   
        k = k + 1
        

        temp[:,k,:,:]    = img_dct[:,(i+16),:,:]
        k = k + 1

        temp[:,k,:,:]    = img_dct[:,(i+32),:,:]
        k = k + 1

    return temp    

def dct_and_rearrange(in_pix,dct_min,dct_max):
    dct_min = dct_min.cuda()
    dct_max = dct_max.cuda()
    b,c,h_inp,w_inp = in_pix.shape
    n_blocks = (h_inp//4)*(w_inp//4) 
    x_pix_2 = torch.zeros_like(in_pix)

    for i in range (c):
        im = in_pix[:,i,:,:].unsqueeze(1)

        im = blockify(im,n_blocks,4)
        im = dct_2d(im)

        img_dct1 = unblockify(im, [h_inp, w_inp], n_blocks, 4)
        x_pix_2[:,i,:,:] = img_dct1.squeeze(1)

       
    x_pix_2 = F.pixel_unshuffle(x_pix_2, 4) 
    x_pix_2 = rearrange_v2(x_pix_2)

    x_pix_2 = (x_pix_2 - dct_min.view(1,48,1,1))/(dct_max.view(1,48,1,1) - dct_min.view(1,48,1,1))
    
    
    
    return x_pix_2



class MSTFusionBlock(nn.Module):
    def __init__(self, dim_imgfeat, kernel_size=3):
        super(MSTFusionBlock, self).__init__()
            
        self.conv_img = nn.Sequential( *[ ResBlock(dim_imgfeat[0], kernel_size) for _ in range(1) ] + [ResBlockAttn(dim_imgfeat[0], kernel_size)])
        self.conv_img1 = nn.Sequential( *[ ResBlock(dim_imgfeat[1], kernel_size) for _ in range(1) ] + [ResBlockAttn(dim_imgfeat[1], kernel_size)])
        self.conv_img2 = nn.Sequential( *[ ResBlock(dim_imgfeat[2], kernel_size) for _ in range(1) ] + [ResBlockAttn(dim_imgfeat[2], kernel_size)]) 
        self.stage_tconv = nn.ConvTranspose2d(dim_imgfeat[1], dim_imgfeat[0], kernel_size=kernel_size, stride=2, padding=(kernel_size//2))
        self.stage_tconv1 = nn.ConvTranspose2d(dim_imgfeat[2], dim_imgfeat[0], kernel_size=kernel_size, stride=4, padding=(kernel_size//2))
        self.msab = MSAB(dim_in=(3*dim_imgfeat[0]), dim_head=dim_imgfeat[0], dim_out=dim_imgfeat[0], heads=3, num_blocks=1)
           
    def forward(self, in_pix, in_pix1,in_pix2):    
        out_pix = self.conv_img(in_pix)
        out_pix1 = self.conv_img1(in_pix1)
        out_pix2 = self.conv_img2(in_pix2)

        out_pix21 = self.stage_tconv1(out_pix2, output_size = in_pix.shape[2:])
        out_pix11 = self.stage_tconv(out_pix1, output_size = in_pix.shape[2:])
        out_pix3 = torch.cat((out_pix11,out_pix21),dim=1)
        out_pix = self.msab(out_pix,out_pix3) + out_pix
        
        #out_pix = self.msab(out_pix, out_dct)
        return out_pix, out_pix1,out_pix2
    

class MSTFusionBlock_crossAttention(nn.Module):
    def __init__(self, dim_imgfeat, kernel_size=3):
        super(MSTFusionBlock_crossAttention, self).__init__()
            
        self.conv_img = nn.Sequential( *[ ResBlock(dim_imgfeat[0], kernel_size) for _ in range(1) ] + [ResBlockAttn(dim_imgfeat[0], kernel_size)])
        self.conv_img1 = nn.Sequential( *[ ResBlock(dim_imgfeat[1], kernel_size) for _ in range(1) ] + [ResBlockAttn(dim_imgfeat[1], kernel_size)])
        self.conv_img2 = nn.Sequential( *[ ResBlock(dim_imgfeat[2], kernel_size) for _ in range(1) ] + [ResBlockAttn(dim_imgfeat[2], kernel_size)]) 
        self.stage_tconv = nn.ConvTranspose2d(dim_imgfeat[1], dim_imgfeat[0], kernel_size=kernel_size, stride=2, padding=(kernel_size//2))
        self.stage_tconv1 = nn.ConvTranspose2d(dim_imgfeat[2], dim_imgfeat[0], kernel_size=kernel_size, stride=4, padding=(kernel_size//2))
        self.msab = MSAB_crossAttention(dim_in=(2*dim_imgfeat[0]), dim_head=dim_imgfeat[0], dim_out=dim_imgfeat[0], dim_cross = dim_imgfeat[2],heads=1, num_blocks=1)
           
    def forward(self, in_pix, in_pix1,in_pix2):    
        out_pix = self.conv_img(in_pix)
        out_pix1 = self.conv_img1(in_pix1)
        out_pix2 = self.conv_img2(in_pix2)

        out_pix21 = self.stage_tconv1(out_pix2, output_size = in_pix.shape[2:])
        out_pix11 = self.stage_tconv(out_pix1, output_size = in_pix.shape[2:])
        out_pix3 = torch.cat((out_pix,out_pix21),dim=1)
        out_pix = self.msab(out_pix3,out_pix11) + out_pix
        
        #out_pix = self.msab(out_pix, out_dct)
        return out_pix, out_pix1,out_pix2

def block(input_channels,output_channels,kernel_size =3):

    return nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels//2, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                                nn.PReLU(output_channels//2),
                                nn.Conv2d(in_channels=output_channels//2, out_channels=output_channels, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                                )



class EDSR_tcfs(nn.Module):
   
    def __init__(self, in_channel=1,out_channel=1,conv=default_conv,n_basicblock=16,dim=[64,64,64],pixel = 1):
        super(EDSR_tcfs, self).__init__()


        out_channel = out_channel
        dic = torch.load('../data/max_min_dct.pth')
        
        kernel_size = 3

        self.pixel = pixel
        
        self.dct_max = dic['max_dct']
        self.dct_min = dic['min_dct']
        # n_basicblock = 16
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.pixel_unshuffle1 = nn.PixelUnshuffle(4)
        # define head module
        self.head_pix_1 = block(12,dim[0])
        self.head_pix_2 =  block(48,dim[1])
        self.head_pix_4 = block(1,dim[2])
        self.thermal_upsample = nn.Upsample(scale_factor=8, mode='bicubic', align_corners=False)
        
        ## Change this to MSTFusionBlock
        self.body = nn.ModuleList([ MSTFusionBlock(dim, kernel_size) for _ in range(n_basicblock) ])
        
        
        
        self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(dim[0], dim[0], 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
        self.upsample = Upsampler(2,dim[0] )
        self.conv_last = nn.Conv2d(dim[0], 1, 3, 1, 1)
        # define tail module
        
        self.tail = nn.Sequential(self.conv_before_upsample, self.upsample, self.conv_last)
        
        # self.unpix_shuffle1 = nn.PixelUnshuffle(4)
        # self.unpix_shuffle = nn.PixelUnshuffle(2)
        # self.tail = conv(dim[0]//4, out_channel, kernel_size)
      
    def forward(self, in_pix,in_thermal):

        """
        in_pix- B x 3 x W x H
        in_dct- B x 48 x W/4 x H/4

        """
        b, c, h_inp, w_inp = in_pix.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        in_pix = F.pad(in_pix, [0, pad_w, 0, pad_h], mode='reflect')

        b, c, h_inp, w_inp = in_pix.shape

        x_pix_1 = self.pixel_unshuffle(in_pix) # B x 12 x W/2 x H/2  

        if self.pixel == 1:
            x_pix_2 = self.pixel_unshuffle1(in_pix) # B x 48 x W/4 x H/4

        else :
            x_pix_2 = dct_and_rearrange(in_pix,self.dct_max,self.dct_min) 


        x_pix_4 = in_thermal

        thermal_upsample = self.thermal_upsample(x_pix_4)
        
        x_pix_1 = self.head_pix_1(x_pix_1)
        x_pix_2 = self.head_pix_2(x_pix_2)
        x_pix_4 = self.head_pix_4(x_pix_4)


        
        for i, layer in enumerate(self.body):
            if i == 0:
                res_pix, x_pix_2,x_pix_4 = layer(x_pix_1, x_pix_2,x_pix_4)
            else:
                res_pix, x_pix_2,x_pix_4 = layer(res_pix, x_pix_2,x_pix_4)
            
        res_pix += x_pix_1


        x_pix = self.tail(res_pix) + thermal_upsample # B x 12 x W/4 x H/4

    
        return x_pix[:,:,:h_inp,:w_inp] 