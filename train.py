import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
from models.EDSR import TSFNet_E2SIFT
from data_utils import Event_to_LoG_Dataset
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import time
from pytorch_msssim import SSIM, ssim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


def train(epoch):
    start = time.time()
    model.train()
    total_loss = 0.0
    running_l1_loss = 0.0
    running_ssim_loss = 0.0
    psnr_sum = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    for batch_index, (voxs, logs, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            voxs = voxs.cuda()
            logs = logs.cuda()
        
        optimizer.zero_grad()
        outputs = model(voxs)

        l1_loss_value = l1_loss(outputs, logs)
        ssim_loss_value = 1 - ssim_loss(outputs, logs)  # Subtract SSIM from 1 to make it a loss

        # Calculate SSIM and MSE for the batch
        batch_ssim = ssim(outputs, logs, data_range=1.0).mean().item()
        batch_mse = mse_check(outputs, logs).item()

        max_pixel_value = 1.0  # Adjust this value based on your data's dynamic range
        psnr_batch = 10 * np.log10((max_pixel_value ** 2) / batch_mse + 1e-8)
        psnr_sum += psnr_batch

        total_ssim += batch_ssim
        total_mse += batch_mse

        # Combine the losses
        loss = l1_loss_value + ssim_loss_value
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        running_l1_loss += l1_loss_value.item()
        running_ssim_loss += ssim_loss_value.item()

        scheduler.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        if batch_index % 100 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.batch_size + len(voxs),
                total_samples=len(train_loader.dataset)
            ))

        # update training loss for each iteration
        # writer.add_scalar('Train/iter loss', loss.item(), n_iter)
        writer.add_scalar('LR/iter', optimizer.param_groups[0]['lr'], n_iter)

    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    
    avg_loss = total_loss / len(train_loader)
    running_l1_loss = running_l1_loss / len(train_loader)
    running_ssim_loss = running_ssim_loss / len(train_loader)
    avg_psnr = psnr_sum / len(train_loader)
    average_mse = total_mse / len(train_loader)
    average_ssim = total_ssim / len(train_loader)

    finish = time.time()
    writer.add_scalar('Train Loss/L1+SSIM', avg_loss, epoch)
    writer.add_scalar('Train Loss/L1', running_l1_loss, epoch)
    writer.add_scalar('Train Loss/SSIM', running_ssim_loss, epoch)
    writer.add_scalar('PSNR/Train', avg_psnr, epoch)
    writer.add_scalar('MSE/Train', average_mse, epoch)
    writer.add_scalar('SSIM/Train', average_ssim, epoch)
    print('epoch {} training time consumed: {:.5f}s, train epoch loss = {:.5f}, PSNR = {:.5f}'.format(epoch, finish - start, avg_loss, avg_psnr))


def eval_training(epoch=0):

    start = time.time()
    model.eval()

    valid_loss = 0.0
    valid_loss_l1 = 0.0
    valid_loss_ssim = 0.0
 
    psnr_sum_val = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for (voxs, logs, _) in valid_loader:

            if torch.cuda.is_available():
                voxs = voxs.cuda()
                logs = logs.cuda()

            outputs = model(voxs)

            
            l1_loss_value = l1_loss(outputs, logs)
            ssim_loss_value = 1 - ssim_loss(outputs, logs)  # Subtract SSIM from 1 to make it a loss
            
            # Calculate SSIM and MSE for the batch
            batch_ssim = ssim(outputs, logs, data_range=1.0).mean().item()
            batch_mse = mse_check(outputs, logs).item()

            max_pixel_value = 1.0  # Adjust this value based on your data's dynamic range
            psnr_batch = 10 * np.log10((max_pixel_value ** 2) / batch_mse + 1e-8)
            psnr_sum_val += psnr_batch

            total_ssim += batch_ssim
            total_mse += batch_mse

            loss = l1_loss_value + ssim_loss_value

            valid_loss += loss.item()
            valid_loss_l1 += l1_loss_value.item()
            valid_loss_ssim += ssim_loss_value.item()
            
        finish = time.time()
        
        avg_loss_val = valid_loss / len(valid_loader)
        valid_loss_l1 = valid_loss_l1 / len(valid_loader)
        valid_loss_ssim = valid_loss_ssim / len(valid_loader)

        avg_psnr_val = psnr_sum_val / len(valid_loader)
        average_mse = total_mse / len(valid_loader)
        average_ssim = total_ssim / len(valid_loader)

        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, PSNR: {:.4f}, MSE: {:.4f}, SSIM: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            avg_loss_val,
            avg_psnr_val,
            average_mse,
            average_ssim,
            finish - start
        ))

        writer.add_scalar('Test Loss/L1+SSIM', avg_loss_val, epoch)
        writer.add_scalar('Test Loss/L1', valid_loss_l1, epoch)
        writer.add_scalar('Test Loss/SSIM', valid_loss_ssim, epoch)
        writer.add_scalar('PSNR/Test', avg_psnr_val, epoch)
        writer.add_scalar('MSE/Test', average_mse, epoch)
        writer.add_scalar('SSIM/Test', average_ssim, epoch)
        return avg_loss_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for TSFNet_E2SIFT")
    parser.add_argument("--vox_path", type=str, nargs="+",
                        help="One or more paths to directories containing training voxel .npz files")
    parser.add_argument("--log_path", type=str, nargs="+",
                        help="One or more paths to directories containing training LoG pyramid .mat files")
    parser.add_argument("--vox_path_valid", type=str, nargs="+",
                        help="One or more paths to directories containing validation voxel .npz files")
    parser.add_argument("--log_path_valid", type=str, nargs="+",
                        help="One or more paths to directories containing validation LoG pyramid .mat files")
    parser.add_argument("--out_path", type=str, default='./logs/',
                        help='Path to output logs')
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
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs")
    parser.add_argument("--init_lr", type=float, default=0.0001,
                        help="Initial learning rate")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help='GPU ID to use for training/validation')
    parser.add_argument("--n_workers", type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument("--id", type=str, default='1',
                        help='Set a unique ID for output logs directory')
    

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    print("\nloading dataset ...")
    train_data_load = Event_to_LoG_Dataset(args.vox_path, args.log_path, 'train', args.vox_clip, args.log_clip, 'sigmoid')
    train_loader = torch.utils.data.DataLoader(train_data_load, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    print(f"Iteration per epoch: {len(train_loader)}")

    valid_data_load = Event_to_LoG_Dataset(args.vox_path_valid, args.log_path_valid, 'valid', args.vox_clip, args.log_clip, 'sigmoid')
    valid_loader = torch.utils.data.DataLoader(valid_data_load, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    print("Validation set samples: ", len(valid_loader))

    model = TSFNet_E2SIFT(args.dct_min, args.dct_max)
    
    num_params = sum(param.numel() for param in model.parameters())
    print(f'Number of model parameters = {num_params} ≈ {num_params/1e6:.2f}M')

    # loss function
    l1_loss = nn.L1Loss()
    ssim_loss = SSIM(channel=4)
    mse_check = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999))

    total_steps = args.epochs * len(train_loader)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )

    # output path
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    date_time = datetime.now().strftime(DATE_FORMAT)
    args.out_path = f'{args.out_path}/{args.id}_{date_time}'
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    ckpt_path = args.out_path + '/ckpt/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if torch.cuda.is_available():
        model.cuda()
        l1_loss.cuda()
        ssim_loss.cuda()

    writer = SummaryWriter(log_dir=ckpt_path)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        loss = eval_training(epoch)

        print('saving weights.')

        if loss < best_loss:
            best_loss = loss

            torch.save(model.state_dict(), os.path.join(ckpt_path, 'best.pth'))
            with open(os.path.join(ckpt_path, 'details.txt'), 'w') as f:
                f.write(f'val_loss = {loss}, epoch = {epoch}, lr = {args.init_lr}, batch_size = {args.batch_size}')
            f.close()

        torch.save(model.state_dict(), os.path.join(ckpt_path, f'{epoch}_best.pth'))
        with open(os.path.join(ckpt_path, f'{epoch}_details.txt'), 'w') as f:
            f.write(f'val_loss = {loss}, epoch = {epoch}, lr = {args.init_lr}, batch_size = {args.batch_size}')
        f.close()

    writer.close()

