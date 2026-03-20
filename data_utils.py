import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from PIL import Image, ImageOps
import os
import glob
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat


def norm_vox_log(arr, clip_min, clip_max, activation):
    # Clip values between clip_min and clip_max
    clipped_array = np.clip(arr, clip_min, clip_max)

    # Normalize values to 0-1
    min_value = np.min(clipped_array)
    max_value = np.max(clipped_array)
    
    if activation == 'sigmoid':
        epsilon = 1e-10
        normalized_array = (clipped_array - min_value) / (max_value - min_value + epsilon)
    if activation == 'tanh':
        normalized_array = (((clipped_array - min_value) / (max_value - min_value))*2)-1
    return clipped_array, normalized_array


def norm_vox_e2vid(vox):
    nonzero_ev = (vox != 0)
    num_nonzeros = nonzero_ev.sum()
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = vox.sum() / num_nonzeros
        stddev = torch.sqrt((vox ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        vox = mask * (vox - mean) / stddev
    return vox


def recon_norm_log(arr, clip_min, clip_max):
    return arr * (clip_max - clip_min) + clip_min


class Event_to_LoG_Dataset(torch.utils.data.Dataset):
    def __init__(self, vox_path, log_path, mode, clip_vox, clip_log, activation):
        self.vox_paths = sorted(
            file
            for path in vox_path
            for file in glob.glob(os.path.join(path, "*.npz"), recursive=True)
        )
        self.log_paths = sorted(
            file
            for path in log_path
            for file in glob.glob(os.path.join(path, "*.mat"), recursive=True)
        )
        self.mode = mode
        self.clip_min_vox, self.clip_max_vox = clip_vox[0], clip_vox[1]
        self.clip_min_log, self.clip_max_log = clip_log[0], clip_log[1]
        self.activation = activation

    def __len__(self):
        return len(self.vox_paths)

    def __getitem__(self, idx):
        vox_path = self.vox_paths[idx]
        log_path = self.log_paths[idx]
        name = os.path.basename(log_path)
        name, ext = os.path.splitext(name)
        
        vox = np.load(vox_path)['arr_0']
        mat_data = loadmat(log_path)
        log_0, log_1, log_2, log_3 = mat_data['log_pyramid']

        vox = norm_vox_e2vid(torch.from_numpy(vox))
        vox = vox.cpu().numpy()
        _, vox = norm_vox_log(vox, self.clip_min_vox, self.clip_max_vox, self.activation)
        _, log_0 = norm_vox_log(log_0, self.clip_min_log, self.clip_max_log, self.activation)
        _, log_1 = norm_vox_log(log_1, self.clip_min_log, self.clip_max_log, self.activation)
        _, log_2 = norm_vox_log(log_2, self.clip_min_log, self.clip_max_log, self.activation)
        _, log_3 = norm_vox_log(log_3, self.clip_min_log, self.clip_max_log, self.activation)

        if self.mode == 'train':
            crop_shape = (5, 160, 160)
            height_start = 0
            width_start = 0
            max_height_start = vox.shape[1] - crop_shape[1]
            max_width_start = vox.shape[2] - crop_shape[2]
            height_start = np.random.randint(0, max_height_start + 1)
            width_start = np.random.randint(0, max_width_start + 1)
            vox = vox[:, height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            vox = torch.from_numpy(vox).float()
        if self.mode == 'valid':
            crop_shape = (5, 160, 160)
            height_start = (vox.shape[1] - crop_shape[1]) // 2
            width_start = (vox.shape[2] - crop_shape[2]) // 2
            vox = vox[:, height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            vox = torch.from_numpy(vox).float()
        
        if self.mode == 'train':
            log_0 = log_0[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            log_1 = log_1[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            log_2 = log_2[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            log_3 = log_3[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]

        if self.mode == 'valid':
            log_0 = log_0[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            log_1 = log_1[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            log_2 = log_2[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            log_3 = log_3[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]

        log_0 = torch.from_numpy(log_0).float().unsqueeze(dim=0)
        log_1 = torch.from_numpy(log_1).float().unsqueeze(dim=0)
        log_2 = torch.from_numpy(log_2).float().unsqueeze(dim=0)
        log_3 = torch.from_numpy(log_3).float().unsqueeze(dim=0)
        
        logs = np.concatenate((log_0, log_1, log_2, log_3), axis=0)

        return vox, logs, name


if __name__ == "__main__":
    vox_path = 'path_to_voxels_dir'
    log_path = 'path_to_LoG_pyramid_dir'

    train_data_load = Event_to_LoG_Dataset(vox_path, log_path, 'train', clip_vox=[-1, 1], clip_log=[-1, 1], activation='sigmoid')
    training_loader = torch.utils.data.DataLoader(train_data_load, batch_size=2, shuffle=False, num_workers=2)
    for vox, log, name in training_loader:
        print (vox.shape, log.shape, name)
        exit(0)

