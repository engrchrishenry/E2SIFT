import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from datetime import datetime
import torch.utils.data
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import data_utils
from scipy.io import loadmat
from models.EDSR import TSFNet_E2SIFT
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from scipy.io import savemat
from pytorch_msssim import ssim as ssimxx
import csv


if __name__ == '__main__':
    path_to_model = '/storage4tb/PycharmProjects/rpg_e2vid/logs2/22_Thursday_09_November_2023_16h_58m_26s/ckpt/best.pth'
    vox_base_path = '/storage/ecd/valid/fix_dur_5_4/vox'
    log_base_path = '/storage/ecd/valid/fix_dur_5_4/log'
    im_base_path = '/storage/ecd/valid/fix_dur_5_4/images'
    out_path = 'xx'
    vox_clip = [-2.5, 2.5]
    log_clip = [-0.2, 0.2]
    # log_clip = [-0.15, 0.15]
    activation = 'sigmoid'
    plot = 1
    save = 1
    num_of_samples = len(glob.glob(f'{vox_base_path}/*.npy'))

    # vox_paths = sorted(glob.glob(f'{vox_base_path}/*.npy'))
    # log_paths = sorted(glob.glob(f'{log_base_path}/*.mat'))
    vox_paths = glob.glob(f'{vox_base_path}/*.npy')
    log_paths = glob.glob(f'{log_base_path}/*.mat')
    
    # model = E2VID()
    model = TSFNet_E2SIFT('/storage4tb/PycharmProjects/rpg_e2vid/dct_min_arr.npy', '/storage4tb/PycharmProjects/rpg_e2vid/dct_max_arr.npy')
    model.load_state_dict(torch.load(path_to_model))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
        
    with torch.no_grad():
        total_mse = 0
        total_psnr = 0
        total_ssim = 0
        c = 1
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        csvfile = open(f'{out_path}/results.csv', 'w', newline='')
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['name', 'mse', 'psnr', 'ssim'])
        # csv_writer.writerow(['calibration', 'dynamic_6dof', 'office_zigzag', 
        # 'poster_6dof', 'shapes_6dof', 'slider_depth'])
        for log_path, vox_path in zip(log_paths[:num_of_samples], vox_paths[:num_of_samples]):
            name = os.path.basename(log_path)
            name, ext = os.path.splitext(name)

            last_underscore_index = name.rfind('_')
            seq_name = name[:last_underscore_index]
            last_underscore_index = seq_name.rfind('_')
            seq_name = seq_name[:last_underscore_index]
            
            vox_original = np.load(vox_path)
            # log = np.load(log_path, allow_pickle=True)[1]
            mat_data = loadmat(log_path)
            logs = mat_data['log_pyramid']
            log_0, log_1, log_2, log_3 = logs
            
            vox_temp = data_utils.norm_vox_e2vid(torch.from_numpy(vox_original))
            vox_temp = vox_temp.cpu().numpy()
            _, vox = data_utils.norm_vox_log(vox_temp, vox_clip[0], vox_clip[1], activation)

            # crop_shape = (5, 160, 160)
            # height_start = (vox.shape[1] - crop_shape[1]) // 2
            # width_start = (vox.shape[2] - crop_shape[2]) // 2
            # vox = vox[:, height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            vox = torch.from_numpy(vox).float().unsqueeze(dim=0)

            # log_0 = log_0[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            # log_1 = log_1[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            # log_2 = log_2[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            # log_3 = log_3[height_start:height_start + crop_shape[1], width_start:width_start + crop_shape[2]]
            
            npy_save_path = f'{out_path}/logs_npy/'
            mat_save_path = f'{out_path}/logs_mat/'
            plots_save_path = f'{out_path}/plots/'
            im_save_path = f'{out_path}/images/'
            if not os.path.exists(npy_save_path):
                os.makedirs(npy_save_path)
            if not os.path.exists(mat_save_path):
                os.makedirs(mat_save_path)
            if not os.path.exists(im_save_path):
                os.makedirs(im_save_path)
            if plot == 1:
                if not os.path.exists(plots_save_path):
                    os.makedirs(plots_save_path)
            
            if torch.cuda.is_available():
                vox = vox.cuda()
            # print ('vox shape', vox.shape)
            # print ('log shape', logs.shape)

            outputs = model(vox)
            outputs = outputs.squeeze().detach().cpu().numpy()

            print ('before recon')
            print ('gt_log_min', logs.min(), 'gt_log_max', logs.max())
            print ('pred_log_min', outputs.min(), 'pred_log_max', outputs.max())

            # outputs = data_utils.recon_norm_log(outputs, log_clip[0], log_clip[1])
            _, logs = data_utils.norm_vox_log(logs, log_clip[0], log_clip[1], 'sigmoid')


            if save == 1:
                np.save(f'{npy_save_path}/{name}.npy', outputs)
                savemat(f'{mat_save_path}/{name}.mat', {'data': outputs})
                shutil.copy(f'{im_base_path}/{name}.png',
                                    f'{im_save_path}/{name}.png')
            
            print ('after recon')
            print ('gt_log_min', logs.min(), 'gt_log_max', logs.max())
            print ('pred_log_min', outputs.min(), 'pred_log_max', outputs.max())
            
            mse = mean_squared_error(logs, outputs)
            # psnr = peak_signal_noise_ratio(logs, outputs, data_range=0.3)
            psnr = peak_signal_noise_ratio(logs, outputs, data_range=1.0)
            temp = 0.0
            for log, output in zip(logs, outputs):
                # ssim = structural_similarity(log, output, data_range=0.3)
                ssim = structural_similarity(log, output, data_range=1.0)
                temp += ssim
                print (ssim, log.shape, output.shape)
            print (temp/len(logs))
                
            print (outputs.shape, logs.shape)
            # ssim = structural_similarity(logs, outputs, channel_axis=0 , data_range=outputs.max()-outputs.min())
            ssim = structural_similarity(logs, outputs, channel_axis=0 , data_range=1.0)
            print (ssim)
            # print (np.expand_dims(outputs, axis=0).shape)
            outputs_torch = torch.from_numpy(outputs).float().unsqueeze(0)
            logs_torch = torch.from_numpy(logs).float().unsqueeze(0)

            # print (outputs_torch.shape, logs_torch.shape)
            batch_ssim = ssimxx(outputs_torch, logs_torch, data_range=1.0).mean().item()
            print (batch_ssim)
            print (outputs_torch.shape, logs_torch.shape)
            # exit(0)

            csv_writer.writerow([name+ext, mse, psnr, ssim])
            total_mse += mse
            total_psnr += psnr
            total_ssim += ssim
            
            vox = vox.detach().cpu().numpy()
            
            if plot == 1:
                fig, axes = plt.subplots(5, len(vox_original), figsize=(12, 7))  # 1 row, 2 columns
                plt.suptitle(f'{name} - PSNR {psnr:.4f} - MSE {mse:.4f} - SSIM {ssim:.4f}')
                for i in range(len(vox_original)):
                    axes[0][i].imshow(vox_original[i, :, :], cmap='gray')  # Use 'gray' colormap for each channel
                    axes[0][i].set_title(f'Channel {i + 1}')
                for i in range(4):
                    axes[1][i].imshow(logs[i], cmap='gray')  # Use 'gray' colormap for each channel
                    axes[1][i].set_title(f'Log {i + 1}')
                    axes[2][i].hist(logs[i])  # Use 'gray' colormap for each channel
                    axes[2][i].set_title(f'Hist {i + 1}')
                    axes[3][i].imshow(outputs[i], cmap='gray')  # Use 'gray' colormap for each channel
                    axes[3][i].set_title(f'Pred {i + 1}')
                    axes[4][i].hist(outputs[i])  # Use 'gray' colormap for each channel
                    axes[4][i].set_title(f'Hist {i + 1}')

                plt.tight_layout()
                plt.savefig(f'{out_path}/plots/{name}.png')
                plt.close()
            
            print(f"{c}/{num_of_samples} {name} MSE: {mse} PSNR: {psnr} dB SSIM: {ssim}")
            c += 1

        average_mse = total_mse / num_of_samples
        average_psnr = total_psnr / num_of_samples
        average_ssim = total_ssim / num_of_samples
        csv_writer.writerow(['Average', average_mse, average_psnr, average_ssim])

        print(f"Average Mean Squared Error (MSE): {average_mse}")
        print(f"Average Peak Signal-to-Noise Ratio (PSNR): {average_psnr} dB")
        print(f"Average Structural Similarity Index (SSIM): {average_ssim}")
    