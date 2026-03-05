import argparse
import os
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from PIL import Image, ImageOps
from joblib import Parallel, delayed
from data_utils import events_to_voxel_grid, myFixedDurationEventReader


def find_nearest_larger_index(arr, value):
    nearest_index = None
    min_difference = float('inf')
    for i, num in enumerate(arr):
        difference = num - value
        if difference > 0 and difference < min_difference:
            nearest_index = i
            min_difference = difference
    return nearest_index


def process_sequence(seq, base_path, out_dir, dur_sec, num_bins, 
                     keypoint_thresh, range_thresh, sd_thresh, num_of_events_th_low,
                     num_of_events_th_high, th_hist, plot):

    event_f_path = f'{base_path}/{seq}/events.txt'
    images_txt_f_path = f'{base_path}/{seq}/images.txt'
    sift = cv2.SIFT_create()

    # print(f'Loading event file for {seq}.')
    start = time.time()
    event_df = pd.read_csv(event_f_path, sep=r"\s+", header=None,
                            names=['t', 'x', 'y', 'pol'],
                            dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                            engine='c', memory_map=True)
    # print(f'Loaded event file for {seq} in {time.time() - start:.2f} sec.')
    
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

        stats = {
            "range_vox": float(range_vox),
            "vox_max": float(vox.max()),
            "vox_min": float(vox.min()),
            "sds": list(np.std(v) for v in vox),
            "avg_sd": float(avg_sd)
        }

        # Save outputs
        np.savez_compressed(f'{out_dir}/vox/{seq}_{im_name}.npz', vox)
        np.savez_compressed(f'{out_dir}/stats/{seq}_{im_name}.npz', stats)
        shutil.copy(im_path, f'{out_dir}/images/{seq}_{im_name}.{im_ext}')

        if plot == 1:
            fig, axes = plt.subplots(3, len(vox), figsize=(12, 7))
            plt.suptitle(f'{len(event_window)} range {range_vox:.3f} max {vox.max():.3f} min {vox.min():.3f}, avg_sd {avg_sd:.3f}, clip_thresh {th_hist}')
            
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
            plt.tight_layout()
            plt.savefig(f'{out_dir}/plots/{seq}_{im_name}.{im_ext}')
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate event voxels from ESIM-generated synthetic events.")

    parser.add_argument("--events_dir", type=str, required=True,
                        help="Path to directory containing ESIM-generated synthetic events")
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--bins', type=int, default=5,
                        help='Number of bins for event voxel generation')
    parser.add_argument('--dur_sec', type=float, default=0.05,
                        help='Event window duration in seconds')
    parser.add_argument("--res", type=str, default='240:180',
                        help="Event camera resolution (e.g., '240:180')")
    parser.add_argument('--events_per_px', type=float, default=0.55,
                        help='Number of events per pixel')
    parser.add_argument('--kp_th', type=int, default=50,
                        help='Keypoint threshold for rejecting blank frames. None to ignore.')
    parser.add_argument('--sd_th', type=float, default=0.1,
                        help='Standard deviation threshold. None to ignore.')
    parser.add_argument('--range_th', type=int, default=20,
                        help='Range (max value - min value) threshold. None to ignore.')
    parser.add_argument('--th_hist', type=float, default=100,
                        help='Clipping threshold for histogram plotting. Value between 0 and 100, e.g., 99.9 means clipping at 99.9 percentile.')
    parser.add_argument('--plot', type=bool, default=True,
                        help='Plot figures. True -> save plots: False -> do not save plots')
    parser.add_argument('--cores', type=int, default=-1,
                        help='Number of cores to use. -1 -> use all cores.')
    
    
    args = parser.parse_args()
    
    events_dir = args.events_dir
    out_dir = args.out_dir
    bins = args.bins
    dur_sec = args.dur_sec
    res = args.res
    events_per_px = args.events_per_px
    kp_th = args.kp_th
    range_th = args.range_th
    sd_th = args.sd_th
    width, height = map(int, args.res.split(":"))
    events_th_low = int(width * height * events_per_px) - 12000
    events_th_high = int(width * height * events_per_px) + 100000
    th_hist = args.th_hist
    plot = args.plot
    cores = args.cores

    out_name = f'{bins}_{dur_sec}_{kp_th}_{events_th_low}_{events_th_high}'
    out_dir = os.path.join(out_dir, out_name)

    # Ensure output directories exist
    for sub in ['images', 'vox', 'plots', "stats"]:
        os.makedirs(f'{out_dir}/{sub}', exist_ok=True)

    with open(f"{out_dir}/params.txt", "w") as f:
        f.write(f'events_dir = {events_dir}\n')
        f.write(f'out_dir = {out_dir}\n')
        f.write(f'num_bins = {bins}\n')
        f.write(f'dur_sec = {dur_sec}\n')
        f.write(f'res = {res}\n')
        f.write(f'events_per_px = {events_per_px}\n')
        f.write(f'kp_th = {kp_th}\n')
        f.write(f'range_thresh = {range_th}\n')
        f.write(f'sd_thresh = {sd_th}\n')
        f.write(f'events_th_low = {events_th_low}\n')
        f.write(f'events_th_high = {events_th_high}\n')
        f.write(f'th_hist = {th_hist}\n')
        f.write(f'plot = {plot}\n')
        f.write(f'cores = {cores}')
    f.close()

    seqs = sorted(os.listdir(events_dir))
    print(f'Found {len(seqs)} sequences to process.')

    Parallel(n_jobs=cores)(
        delayed(process_sequence)(seq, events_dir, out_dir, dur_sec, bins, 
                                  kp_th, range_th, sd_th, events_th_low,
                                  events_th_high, th_hist, plot)
        for seq in tqdm(seqs)
    )

