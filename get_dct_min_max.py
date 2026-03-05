import argparse
import sys
import os
import random
from models.torch_dct import dct_2d
from models.EDSR import blockify, unblockify, rearrange_v2
from data_utils import norm_vox_log
import numpy as np
import torch
import torch.nn.functional as F


def dct_and_rearrange(in_pix):
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

    return x_pix_2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get min and max for DCT normalization in TSFNet_E2SIFT model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the directory containing event voxels')
    parser.add_argument("--vox_clip", type=float, nargs=2, metavar=('min', 'max'), required=True,
                        help='Min and max clipping value for event voxels')
    parser.add_argument('--num_voxels', type=int, required=True,
                        help='Number of event voxels to process')
    parser.add_argument('--out_path', type=str, default='output/dct_norm',
                        help='Path to the output directory')

    args = parser.parse_args()

    data_dir = args.data_dir
    num_voxels = args.num_voxels # 4000 # 20000 10000
    out_path = args.out_path
    vox_clip_min = args.vox_clip[0]
    vox_clip_max = args.vox_clip[1]

    os.makedirs(args.out_path, exist_ok=True)

    vox_paths = os.listdir(data_dir)
    random.shuffle(vox_paths)
    tensor_list = []
    for i, vox_path in enumerate(vox_paths[:num_voxels]):
        vox = np.load(f'{data_dir}/{vox_path}')['arr_0']
        _, vox = norm_vox_log(vox, vox_clip_min, vox_clip_max, 'sigmoid')

        vox = torch.from_numpy(vox).float().unsqueeze(dim=0)
        tensor_list.append(vox)
        if i % 100 == 0:
            print (i, '/', len(vox_paths[:num_voxels]))
        
    tensor_list = torch.cat(tensor_list, dim=0)
    
    dct = dct_and_rearrange(tensor_list)
    max_list = []
    min_list = []
    for i in range(80):
        max_list.append(dct[:, i, :, :].max())
        min_list.append(dct[:, i, :, :].min())    
        # print('dct.max(), dct.min()', dct[:, i, :, :].max(), dct[:, i, :, :].min())
    
    print (len(max_list), len(min_list))
    max_arr = np.array(max_list)
    min_arr = np.array(min_list)
    print(max_arr.shape, min_arr.shape)

    np.save(f'{out_path}/dct_min.npy', min_arr)
    np.save(f'{out_path}/dct_max.npy', max_arr)
    print(f'Saved DCT min array to {out_path}/dct_min.npy')
    print(f'Saved DCT max array to {out_path}/dct_max.npy')

