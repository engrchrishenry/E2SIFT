import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from datetime import datetime
from model.model_chris import E2VID
from models_biren.EDSR import TSFNet, TSFNet_Chris
from utils.loading_utils import load_model
from data_utils import Event_Camera_Dataset, Event_Camera_Dataset_LoG, Event_Camera_Dataset_single
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import time
from pytorch_msssim import SSIM, ssim
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import matplotlib.pyplot as plt

# Define a function to calculate SSIM and MSE
def calculate_ssim_mse(output, target):
    ssim = 0.0
    mse = 0.0
    for i in range(output.size(0)):
        # Convert tensors to numpy arrays
        output_i = output[i].cpu().detach().numpy()
        target_i = target[i].cpu().detach().numpy()
        
        # Calculate SSIM for each channel separately
        ssim_i = 0.0
        for channel in range(output_i.shape[0]):
            ssim_i += compare_ssim(target_i[channel], output_i[channel], data_range=1)

        ssim_i /= output_i.shape[0]  # Average SSIM over channels
        ssim += ssim_i

        # Calculate MSE
        mse += ((target_i - output_i) ** 2).mean()

    return ssim / output.size(0), mse / output.size(0)


def train(epoch):
    start = time.time()
    model.train()
    total_loss = 0.0
    running_l1_loss = 0.0
    running_ssim_loss = 0.0
    psnr_sum = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    for batch_index, (voxs, dogs) in enumerate(train_loader):
        # dog_0, _, _, _ = dogs
        plt.figure(1);plt.imshow(voxs[1,4,:,:].data.cpu(), cmap="gray");plt.figure(2);plt.imshow(voxs[1,0,:,:].data.cpu(), cmap="gray")
        hist, bins = np.histogram(voxs[1,4,:,:], bins=256);plt.figure(3);plt.plot(bins[:-1], hist);plt.grid()
        if torch.cuda.is_available():
            voxs = voxs.cuda()
            # dog_0 = dog_0.cuda()
            dogs = dogs.cuda()
        
        optimizer.zero_grad()
        outputs = model(voxs)

        # psnr_batch = psnr(dogs, outputs, max_value=1.0)
        # psnr_sum += psnr_batch


        # loss = criterion(outputs, dog_0)
        l1_loss_value = l1_loss(outputs, dogs)
        ssim_loss_value = 1 - ssim_loss(outputs, dogs)  # Subtract SSIM from 1 to make it a loss

        # Calculate SSIM and MSE for the batch
        # batch_ssim, batch_mse = calculate_ssim_mse(outputs, dogs)
        batch_ssim = ssim(outputs, dogs, data_range=1.0).mean().item()
        batch_mse = mse_check(outputs, dogs).item()

        max_pixel_value = 1.0  # Adjust this value based on your data's dynamic range
        psnr_batch = 10 * np.log10((max_pixel_value ** 2) / batch_mse)
        psnr_sum += psnr_batch

        total_ssim += batch_ssim
        total_mse += batch_mse

        # Combine the losses using a weighted sum (adjust weights as needed)
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
                trained_samples=batch_index * opt.batch_size + len(voxs),
                total_samples=len(train_loader.dataset)
            ))

        #update training loss for each iteration
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
    # writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
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
        for (voxs, dogs) in valid_loader:
            # dog_0, _, _, _ = dogs

            if torch.cuda.is_available():
                voxs = voxs.cuda()
                # dog_0 = dog_0.cuda()
                dogs = dogs.cuda()

            outputs = model(voxs)

            # psnr_batch = psnr(dogs, outputs, max_value=1.0)
            # psnr_sum_val += psnr_batch
            
            # loss = criterion(outputs, dog_0)
            l1_loss_value = l1_loss(outputs, dogs)
            ssim_loss_value = 1 - ssim_loss(outputs, dogs)  # Subtract SSIM from 1 to make it a loss
            # Calculate SSIM and MSE for the batch
            # batch_ssim, batch_mse = calculate_ssim_mse(outputs, dogs)
            batch_ssim = ssim(outputs, dogs, data_range=1.0).mean().item()
            batch_mse = mse_check(outputs, dogs).item()

            max_pixel_value = 1.0  # Adjust this value based on your data's dynamic range
            psnr_batch = 10 * np.log10((max_pixel_value ** 2) / batch_mse)
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


def psnr(original, reconstructed, max_value=1.0):
    mse = torch.mean((original - reconstructed) ** 2)
    psnr_value = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr_value.item()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="E2VID Training")
    parser.add_argument('--use_pretrain', type=str, default=False, help='use pretrained model weights')
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--end_epoch", type=int, default=100, help="number of epochs")
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--outf", type=str, default='./logs/', help='path log files')
    parser.add_argument("--vox_path", type=str, default='/storage4tb/PycharmProjects/rpg_e2vid/output/ESIM/vox')
    parser.add_argument("--dog_path", type=str, default='/storage4tb/PycharmProjects/rpg_e2vid/output/LoG_Pyramid')
    parser.add_argument("--vox_path_valid", type=str, default='/storage4tb/PycharmProjects/rpg_e2vid/output/Event_Camera_Dataset_fix_dur_valid_for_esim_training/vox')
    parser.add_argument("--dog_path_valid", type=str, default='/storage4tb/PycharmProjects/rpg_e2vid/output/LoG_Pyramid_valid')
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
    opt = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    id = '11'
    
    print("\nloading dataset ...")
    # train_data_load = Event_Camera_Dataset(opt.vox_path, opt.dog_path, 'train', [-4, 4], [-8, 8])
    train_data_load = Event_Camera_Dataset_LoG(opt.vox_path, opt.dog_path, 'train', [-5, 5], [-0.3, 0.3])
    # train_data_load = Event_Camera_Dataset_single(opt.vox_path, opt.dog_path, 'train', [-4, 4], [-8, 8])
    train_loader = torch.utils.data.DataLoader(train_data_load, batch_size=opt.batch_size, shuffle=True, num_workers=3)
    print(f"Iteration per epoch: {len(train_loader)}")
    # valid_data_load = Event_Camera_Dataset(opt.vox_path_valid, opt.dog_path_valid, 'valid', [-4, 4], [-8, 8])
    valid_data_load = Event_Camera_Dataset_LoG(opt.vox_path_valid, opt.dog_path_valid, 'valid', [-5, 5], [-0.3, 0.3])
    # valid_data_load = Event_Camera_Dataset_single(opt.vox_path_valid, opt.dog_path_valid, 'valid', [-4, 4], [-8, 8])
    valid_loader = torch.utils.data.DataLoader(valid_data_load, batch_size=opt.batch_size, shuffle=False, num_workers=3)
    print("Validation set samples: ", len(valid_loader))

    # model
    # Add how to load model from pretrained weights
    if opt.use_pretrain:
        model = load_model(opt.pretrained_model_path)
    else:
        # model = E2VID()
        model = TSFNet_Chris()
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    # loss function
    l1_loss = nn.L1Loss()
    ssim_loss = SSIM(channel=4)
    mse_check = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))

    # total_steps = (opt.end_epoch + 1) * len(train_loader)
    # total_steps_per_epoch = len(train_loader.dataset) // opt.batch_size
    total_steps_per_epoch = len(train_loader)
    T_max_fraction = 1.0
    T_max = int(total_steps_per_epoch * T_max_fraction)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)

    # output path
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    date_time = datetime.now().strftime(DATE_FORMAT)
    opt.outf = f'{opt.outf}/{id}_{date_time}'
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    ckpt_path = opt.outf + '/ckpt/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if torch.cuda.is_available():
        model.cuda()
        l1_loss.cuda()
        ssim_loss.cuda()

    writer = SummaryWriter(log_dir=ckpt_path)

    # input_tensor = torch.Tensor(1, 5, 160, 160)
    # if torch.cuda.is_available():
    #     input_tensor = input_tensor.cuda()
    # writer.add_graph(model, input_tensor)

    best_loss = float('inf')
    for epoch in range(1, opt.end_epoch + 1):
        train(epoch)
        loss = eval_training(epoch)

        print('saving weights.')

        if loss < best_loss:
            best_loss = loss
            # weights_path = checkpoint_path.format(net='best', epoch='', acc='')

            torch.save(model.state_dict(), os.path.join(ckpt_path, 'best.pth'))
            with open(os.path.join(ckpt_path, 'details.txt'), 'w') as f:
                f.write(f'val_loss = {loss}, epoch = {epoch}, lr = {opt.init_lr}, batch_size = {opt.batch_size}')
            f.close()

        torch.save(model.state_dict(), os.path.join(ckpt_path, f'{epoch}_best.pth'))
        with open(os.path.join(ckpt_path, f'{epoch}_details.txt'), 'w') as f:
            f.write(f'val_loss = {loss}, epoch = {epoch}, lr = {opt.init_lr}, batch_size = {opt.batch_size}')
        f.close()

    writer.close()