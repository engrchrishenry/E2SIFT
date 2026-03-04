from base.base_model_chris import BaseModel
import torch.nn as nn
import torch
from model.unet import UNet, UNetRecurrent
from os.path import join
from model.submodules import ConvLSTM, ResidualBlock, ConvLayer, UpsampleConvLayer, TransposedConvLayer


class BaseE2VID(BaseModel):
    def __init__(self):
        super().__init__()

        self.num_bins = 5  # number of bins in the voxel grid event tensor
        self.skip_type = 'sum'
        self.num_encoders = 4
        self.base_num_channels = 32
        self.num_residual_blocks = 2
        self.norm = None
        self.use_upsample_conv = True


class E2VID(BaseE2VID):
    def __init__(self):
        super(E2VID, self).__init__()

        self.unet = UNet(num_input_channels=self.num_bins,
                         num_output_channels=4,
                         skip_type=self.skip_type,
                         activation='sigmoid', # tanh sigmoid
                         num_encoders=self.num_encoders,
                         base_num_channels=self.base_num_channels,
                         num_residual_blocks=self.num_residual_blocks,
                         norm=self.norm,
                         use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states=None):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
        return self.unet.forward(event_tensor)


class E2VIDRecurrent(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self):
        super(E2VIDRecurrent, self).__init__()

        self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unetrecurrent = UNetRecurrent(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)
        return img_pred, states
