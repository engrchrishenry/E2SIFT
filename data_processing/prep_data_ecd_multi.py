import os
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import pysift

from PIL import Image, ImageOps
from joblib import Parallel, delayed
from scipy.signal import max_len_seq
from scipy import fftpack
from utils.event_readers import myFixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid
from data_utils import get_duration_s

def find_nearest_larger_index(arr, value):
    nearest_index = None
    min_difference = float('inf')
    for i, num in enumerate(arr):
        difference = num - value
        if difference > 0 and difference < min_difference:
            nearest_index = i
            min_difference = difference
    return nearest_index

def process_sequence(seq, base_path, out_path, dur_sec, num_bins, num_event_per_pixel,
                     keypoint_thresh, range_thresh, sd_thresh, num_of_events_th_low,
                     num_of_events_th_high, th_hist, plot):

    event_f_path = f'{base_path}/{seq}/events.txt'
    images_txt_f_path = f'{base_path}/{seq}/images.txt'
    sift = cv2.SIFT_create()

    try:
        print(f'Loading event file for {seq}.')
        start = time.time()
        event_df = pd.read_csv(event_f_path, delim_whitespace=True, header=None,
                               names=['t', 'x', 'y', 'pol'],
                               dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                               engine='c', memory_map=True)
        print(f'Loaded event file for {seq} in {time.time() - start:.2f} sec.')
    except Exception as e:
        print(f'Failed to load {seq}: {e}')
        return

    lines = open(images_txt_f_path, 'r').readlines()
    ts_lines = [float(temp.split(' ')[0]) for temp in lines]
    if not ts_lines:
        return

    first_frame_ts = ts_lines[0]
    last_frame_ts = ts_lines[-1]

    for i, ts in enumerate(np.arange(first_frame_ts + dur_sec, last_frame_ts, dur_sec)):
        gt_im_index = find_nearest_larger_index(ts_lines, ts)
        if gt_im_index is None:
            continue
        if i == 0:
            start_frame_index = 0
        else:
            start_frame_index = find_nearest_larger_index(ts_lines, ts - dur_sec)

        start_time, _ = lines[start_frame_index].split(' ')
        end_time, gt_frame_name = lines[gt_im_index].split(' ')
        gt_frame_name = gt_frame_name.strip()
        start_time = float(start_time)
        end_time = float(end_time)
        im_name, im_ext = gt_frame_name.replace('images/', '').split('.')

        im_path = f'{base_path}/{seq}/{gt_frame_name}'
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue

        keypoints, descriptors = sift.detectAndCompute(im, None)
        if keypoint_thresh is not None and len(keypoints) < keypoint_thresh:
            continue

        width, height = im.shape[1], im.shape[0]
        num_events = int(width * height * num_event_per_pixel)

        event_window = myFixedDurationEventReader(event_df, end_time=end_time, start_time=start_time)
        if event_window.size == 0:
            continue
        
        if num_of_events_th_low is not None or num_of_events_th_high is not None:
            if len(event_window) < num_of_events_th_low or len(event_window) > num_of_events_th_high:
                continue

        vox = events_to_voxel_grid(event_window, num_bins=num_bins, width=width, height=height)
        range_vox = vox.max() - vox.min()
        avg_sd = 0
        sd_indicator = 0

        for temp in vox:
            sd_temp = np.std(temp)
            if sd_thresh is not None and sd_temp < sd_thresh:
                sd_indicator = 1
                break
            avg_sd += sd_temp
        avg_sd /= len(vox)

        if range_thresh is not None and range_vox > range_thresh:
            continue
        if sd_thresh is not None and sd_indicator == 1:
            continue

        dog = pysift.computeDoGImages(im, num_intervals=2)

        # Save outputs
        np.save(f'{out_path}/vox/{seq}_{im_name}.npy', vox)
        np.save(f'{out_path}/dog/{seq}_{im_name}.npy', dog)
        shutil.copy(im_path, f'{out_path}/images/{seq}_{im_name}.{im_ext}')

        if plot == 1:
            fig, axes = plt.subplots(4, len(vox), figsize=(13, 9))
            plt.suptitle(f'{len(event_window)}/{num_events} range {range_vox:.3f} max {vox.max():.3f} min {vox.min():.3f}, avg_sd {avg_sd:.3f}, clip_thresh {th_hist}')
            
            percentile_max = np.percentile(vox.flatten(), th_hist)
            percentile_min = np.percentile(vox.flatten(), 100-th_hist)

            for j in range(len(vox)):
                img = axes[0][j].imshow(vox[j, :, :], cmap='gray')
                axes[0][j].set_title(f'Vox Bin {j + 1}')
                plt.colorbar(img, ax=axes[0][j])
                axes[1][j].hist(vox[j, :, :].ravel())
                axes[1][j].set_title(f'SD {np.std(vox[j]):.3f}')
                img = axes[2][j].imshow(np.clip(vox[j], percentile_min, percentile_max), cmap='gray')
                axes[2][j].set_title(f'Vox Bin Clipped {j + 1}')
                plt.colorbar(img, ax=axes[2][j])
            for j in range(4):
                img = axes[3][j].imshow(dog[j], cmap='gray')
                axes[3][j].set_title(f'Dog {j + 1}')
                plt.colorbar(img, ax=axes[3][j])
            
            plt.tight_layout()
            plt.savefig(f'{out_path}/plots/{seq}_{im_name}.{im_ext}')
            plt.close()

if __name__ == "__main__":
    base_path = '/storage4tb/PycharmProjects/rpg_e2vid/data/Event_Camera_Dataset/valid'
    out_path = '/storage4tb/PycharmProjects/rpg_e2vid/output/ecd_new_ev_th/'
    mode = 'fix_dur'
    split_name = 'valid'
    num_bins = 5
    num_of_frames = 4
    num_event_per_pixel = 0.55
    dur_sec = 0.005 # in sec
    keypoint_thresh = 50 # None to ignore, 50
    range_thresh = None # None to ignore, 20
    sd_thresh = None # None to ignore, 0.1
    num_of_events_th_low = 12000 # None to ignore, 12000
    num_of_events_th_high = 100000 # None to ignore, 100000
    th_hist = 99.9
    plot = 1
    n_jobs = -1  # Use all available cores

    print('Using fixed duration mode')
    out_name = f'{split_name}/{mode}_{num_bins}_{num_of_frames}_{dur_sec}'
    out_path = os.path.join(out_path, out_name)

    # Ensure output directories exist
    for sub in ['dog', 'images', 'vox', 'plots']:
        os.makedirs(f'{out_path}/{sub}', exist_ok=True)

    seqs = sorted(os.listdir(base_path))
    print(f'Found {len(seqs)} sequences to process.')

    Parallel(n_jobs=n_jobs)(
        delayed(process_sequence)(seq, base_path, out_path, dur_sec, num_bins, num_event_per_pixel,
                                  keypoint_thresh, range_thresh, sd_thresh, num_of_events_th_low,
                                  num_of_events_th_high, th_hist, plot)
        for seq in seqs
    )

