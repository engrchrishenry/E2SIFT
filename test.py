import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from models.EDSR import TSFNet_E2SIFT
from data_utils import Event_Camera_Dataset_LoG, recon_norm_log
import torch.utils.data
from pytorch_msssim import SSIM, ssim
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing script for TSFNet_E2SIFT")
    parser.add_argument("--vox_path", type=str, nargs="+",
                        help="One or more paths to directories containing training voxel .npz files")
    parser.add_argument("--log_path", type=str, nargs="+",
                        help="One or more paths to directories containing training LoG pyramid .mat files")
    parser.add_argument("--weights", type=str, 
                        help='Path to trained weights')
    parser.add_argument("--out_path", type=str,
                        help='Path to output predicted LoG pyramid')
    parser.add_argument("--vox_clip", type=float, nargs=2, metavar=('min', 'max'),
                        help='Min and max clipping value for event voxels')
    parser.add_argument("--log_clip", type=float, nargs=2, metavar=('min', 'max'),
                        help='Min and max clipping value for LoG pyramid')
    parser.add_argument("--dct_min", type=str, 
                        help='Path to dct_min.npy (generated via get_dct_min_max.py)')
    parser.add_argument("--dct_max", type=str, 
                        help='Path to dct_max.npy (generated via get_dct_min_max.py)')
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--n_workers", type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--plot', action="store_true",
                        help='Save plots. If not set, only .mat files will be saved')

    args = parser.parse_args()

    test_data_load = Event_Camera_Dataset_LoG(args.vox_path, args.log_path, 'valid', args.vox_clip, args.log_clip, 'sigmoid')
    test_loader = torch.utils.data.DataLoader(test_data_load, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    print("Test batches: ", len(test_loader))

    model = TSFNet_E2SIFT(args.dct_min, args.dct_max)
    model.load_state_dict(torch.load(args.weights))
    ssim_loss = SSIM(channel=4)
    mse_check = nn.MSELoss()
    if torch.cuda.is_available():
        model.cuda()
        ssim_loss.cuda()
    model.eval()

    if not os.path.exists(f'{args.out_path}/pred_log'):
        os.makedirs(f'{args.out_path}/pred_log')
    if not os.path.exists(f'{args.out_path}/plots'):
        os.makedirs(f'{args.out_path}/plots')

    psnr_sum_val = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for idx, (voxs, logs, names) in enumerate(test_loader):
            print (f'Processing batch: {idx+1}/{len(test_loader)}')
            
            if torch.cuda.is_available():
                voxs = voxs.cuda()
                logs = logs.cuda()

            outputs = model(voxs)
            
            batch_ssim = ssim(outputs, logs, data_range=1.0).mean().item()
            batch_mse = mse_check(outputs, logs).item()

            max_pixel_value = 1.0
            psnr_batch = 10 * np.log10((max_pixel_value ** 2) / batch_mse)
            psnr_sum_val += psnr_batch

            total_ssim += batch_ssim
            total_mse += batch_mse
            
            outputs = outputs.squeeze().detach().cpu().numpy()
            outputs = recon_norm_log(outputs, args.log_clip[0], args.log_clip[1])
            logs = logs.squeeze().detach().cpu().numpy()
            logs = recon_norm_log(logs, args.log_clip[0], args.log_clip[1])
            if outputs.ndim == 3: # for batch size of 1, outputs will have shape (4, H, W) instead of (B, 4, H, W)
                savemat(f'{args.out_path}/pred_log/{names[0]}.mat', {'log_pyramid': outputs})
                if args.plot:
                    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
                    for i in range(4):
                        axes[0, i].imshow(logs[i], cmap="gray")
                        axes[0, i].set_title(f"GT LoG Channel {i}")
                        axes[0, i].axis("off")
                        axes[1, i].imshow(outputs[i], cmap="gray")
                        axes[1, i].set_title(f"Pred LoG Channel {i}")
                        axes[1, i].axis("off")
                    fig.suptitle(names[0], fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f"{args.out_path}/plots/{names[0]}.png")
                    plt.close()
            else: # for batch size greater than 1, outputs will have shape (B, 4, H, W)
                for idx, pred_log in enumerate(outputs):
                    savemat(f'{args.out_path}/pred_log/{names[idx]}.mat', {'log_pyramid': pred_log})
                    if args.plot:
                        fig, axes = plt.subplots(2, 4, figsize=(8, 4))
                        for i in range(4):
                            axes[0, i].imshow(logs[idx][i], cmap="gray")
                            axes[0, i].set_title(f"GT LoG Channel {i}")
                            axes[0, i].axis("off")
                            axes[1, i].imshow(pred_log[i], cmap="gray")
                            axes[1, i].set_title(f"Pred LoG Channel {i}")
                            axes[1, i].axis("off")
                        fig.suptitle(names[idx], fontsize=14)
                        plt.tight_layout()
                        plt.savefig(f"{args.out_path}/plots/{names[idx]}.png")
                        plt.close()

        avg_psnr_val = psnr_sum_val / len(test_loader)
        average_mse = total_mse / len(test_loader)
        average_ssim = total_ssim / len(test_loader)
        
        print('Test set: PSNR: {:.7f}, MSE: {:.7f}, SSIM: {:.7f}'.format(
            avg_psnr_val,
            average_mse,
            average_ssim
        ))

        with open(f"{args.out_path}/results.txt", "w") as f:
            f.write(f"Test set: PSNR: {avg_psnr_val:.7f}, MSE: {average_mse:.7f}, SSIM: {average_ssim:.7f}")

