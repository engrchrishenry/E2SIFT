import numpy as np
from PIL import Image, ImageOps
import os
# from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader, myFixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid
import pysift
import cv2
import pandas as pd
import time
import shutil
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_nearest_larger_index(arr, value):
    # Initialize variables to store the nearest index and the minimum difference
    nearest_index = None
    min_difference = float('inf')

    # Iterate through the list and find the nearest integer greater than the specified value
    for i, num in enumerate(arr):
        difference = num - value
        if difference > 0 and difference < min_difference:
            nearest_index = i
            min_difference = difference

    return nearest_index


def get_duration_esim(ts_file_frame_path, events_path, start_frame, num_of_frames):
    f = open(ts_file_frame_path, 'r')
    lines = f.readlines()[:len(os.listdir(events_path))] # because sometimes no. of event files are less than the upsampled frames.

    prev_time = None
    durations_s = []
    durations_micro_s = []
    for i in range(start_frame, len(lines), num_of_frames): # starts with 1 to skip 1 line in events.txt file
        line = lines[i].replace('\n', '')
        time = float(line)
        if prev_time == None:
            prev_time = time
            continue
        duration_s = time - prev_time
        im_name = f'{i:08d}.png'
        im_name_prev = f'{i-num_of_frames:010d}.png'
        list_of_event_files = [f'{x:010d}.npz' for x in range(i-num_of_frames, i)]
        durations_s.append([duration_s, im_name, list_of_event_files])
        durations_micro_s.append([duration_s*1e+9, im_name, list_of_event_files])
        prev_time = time
    return durations_s, durations_micro_s #  # durations_s/durations_ms gives [duration, start_time, end_time, im_name]]


def process_sequence(seq, events_path, upsamp_frames_path, out_path, dur_sec, num_bins, num_event_per_pixel,
                     keypoint_thresh, range_thresh, sd_thresh, num_of_events_th_low, num_of_events_th_high, th_hist, plot, save_files):
    sift = cv2.SIFT_create()
    ts_file_frame_path = f'{upsamp_frames_path}/{seq}/timestamps.txt'
    event_files = os.listdir(f'{events_path}/{seq}')
    lines = open(ts_file_frame_path, 'r').readlines()[:len(event_files)]
    lines = [float(temp) for temp in lines]
    if not lines:
        return

    first_frame_ts = lines[0]
    last_frame_ts = lines[-1]

    for i, ts in enumerate(np.arange(first_frame_ts + dur_sec, last_frame_ts, dur_sec)):
        gt_im_index = find_nearest_larger_index(lines, ts)
        if gt_im_index is None:
            continue
        if i == 0:
            start_frame_index = 0
        else:
            start_frame_index = find_nearest_larger_index(lines, ts - dur_sec)

        if gt_im_index is None or start_frame_index is None:
            continue

        gt_im_name = f'{gt_im_index:08d}.png'
        im_name, im_ext = gt_im_name.split('.')
        try:
            im = cv2.imread(f'{upsamp_frames_path}/{seq}/imgs/{gt_im_name}', cv2.IMREAD_GRAYSCALE)
        except:
            continue

        if im is None:
            continue

        keypoints, descriptors = sift.detectAndCompute(im, None)
        if keypoint_thresh is not None and len(keypoints) < keypoint_thresh:
            continue

        width, height = im.shape[1], im.shape[0]
        num_events = int(height * width * num_event_per_pixel)

        dog = pysift.computeDoGImages(im, num_intervals=2)

        # Aggregate events
        x, y, t, p = [], [], [], []
        for event_name in range(start_frame_index, gt_im_index + 1):
            ev_file = f'{event_name:010d}.npz'
            ev_path = f'{events_path}/{seq}/{ev_file}'
            if not os.path.exists(ev_path):
                continue
            ev = np.load(ev_path)
            x.append(ev['x'])
            y.append(ev['y'])
            t.append(ev['t'].astype(np.float32) / 1e9)
            p.append(ev['p'])

        if not x:
            continue

        x = np.concatenate(x)
        y = np.concatenate(y)
        t = np.concatenate(t)
        p = np.concatenate(p)

        if ((num_of_events_th_low is not None and len(t) < num_of_events_th_low) or
            (num_of_events_th_high is not None and len(t) > num_of_events_th_high)):
            continue

        event_window = np.stack([t, x, y, p], axis=1)
        vox = events_to_voxel_grid(event_window, num_bins=num_bins, width=width, height=height)

        # Filtering by std and range
        range_vox = vox.max() - vox.min()
        avg_sd = np.mean([np.std(v) for v in vox])
        if (range_thresh is not None and range_vox > range_thresh):
            continue
        if (sd_thresh is not None and any(np.std(v) < sd_thresh for v in vox)):
            continue
        
        stats = {
            "range_vox": float(range_vox),
            "vox_max": float(vox.max()),
            "vox_min": float(vox.min()),
            "sds": list(np.std(v) for v in vox),
            "avg_sd": float(avg_sd),
            "num_events": float(len(x))
        }

        # Save
        if save_files == 1:
            # np.save(f'{out_path}/vox/{seq}_{im_name}.npy', vox)
            # np.save(f'{out_path}/dog/{seq}_{im_name}.npy', dog)
            np.savez_compressed(f'{out_path}/vox/{seq}_{im_name}', vox=vox)
            # np.savez_compressed(f'{out_path}/dog/{seq}_{im_name}', dog=dog)
            np.savez_compressed(f'{out_path}/stats/{seq}_{im_name}', stats=stats)
            shutil.copy(f'{upsamp_frames_path}/{seq}/imgs/{gt_im_name}',
                        f'{out_path}/images/{seq}_{im_name}.{im_ext}')
        
        if plot == 1:
            fig, axes = plt.subplots(4, len(vox), figsize=(13, 9))
            plt.suptitle(f'{len(x)}/{num_events} range {range_vox:.3f} max {vox.max():.3f} min {vox.min():.3f}, avg_sd {avg_sd:.3f}, clip_thresh {th_hist}')
            
            percentile_max = np.percentile(vox.flatten(), th_hist)
            percentile_min = np.percentile(vox.flatten(), 100-th_hist)
            
            for j in range(len(vox)):
                img = axes[0][j].imshow(vox[j], cmap='gray')
                plt.colorbar(img, ax=axes[0][j])
                axes[0][j].set_title(f'Vox Bin {j + 1}')
                axes[1][j].hist(vox[j].ravel())
                axes[1][j].set_title(f'SD {np.std(vox[j]):.3f}')
                img = axes[2][j].imshow(np.clip(vox[j], percentile_min, percentile_max), cmap='gray')
                axes[2][j].set_title(f'Vox Bin Clipped {j + 1}')
                plt.colorbar(img, ax=axes[2][j])
            for j in range(min(len(dog), 4)):
                img = axes[3][j].imshow(dog[j], cmap='gray')
                axes[3][j].set_title(f'DoG {j+1}')
                plt.colorbar(img, ax=axes[3][j])
            plt.tight_layout()
            plt.savefig(f'{out_path}/plots/{seq}_{im_name}.{im_ext}')
            plt.close()

if __name__ == "__main__":
    events_path = '/storage4tb/PycharmProjects/rpg_vid2e/example/events_reds_120fps/val'
    upsamp_frames_path = '/storage4tb/PycharmProjects/Datasets/reds_120fps_resized_ts_533_300/val'
    # events_path = '/storage4tb/PycharmProjects/rpg_vid2e/example/events_degraded_affine_0.2'
    # upsamp_frames_path = 'output/vimeo_upsampled_degraded/affine/'
    out_path = 'output/updated/esim_reds_filtered/val'
    mode = 'fix_dur_sec_events' # fix_dur fix_events fix_dur_sec
    split_name = '' # train valid'
    num_bins = 5
    num_event_per_pixel = 0.55
    dur_sec = 0.005 # in sec 0.25
    keypoint_thresh = 50 # None to ignore, 50
    range_thresh = None # None to ignore, 20
    sd_thresh = None # None to ignore, 0.1
    num_of_events_th_low = 70000 # None to ignore, 1ms-640x360 90000 5ms-640x360 100000 5ms-533x300 70000
    num_of_events_th_high = 300000 # None to ignore, 1ms-640x360 400000 5ms-640x360 400000 5ms-533x300 300000
    th_hist = 99.9
    plot = 1
    save_files = 1
    n_cores = 18 # -1

    out_name = f'{split_name}/{num_bins}_{num_event_per_pixel}_{dur_sec}_{keypoint_thresh}_{num_of_events_th_low}_{num_of_events_th_high}'
    out_path = f'{out_path}/{out_name}'

    for sub in ["dog", "images", "vox", "plots", "stats"]:
        os.makedirs(f"{out_path}/{sub}", exist_ok=True)
    
    with open(f"{out_path}/settings.txt", "w") as f:
        f.write(f'events_path = {events_path}\n')
        f.write(f'upsamp_frames_path = {upsamp_frames_path}\n')
        f.write(f'out_path = {out_path}\n')
        f.write(f'mode = {mode}\n')
        f.write(f'num_bins = {num_bins}\n')
        f.write(f'num_event_per_pixel = {num_event_per_pixel}\n')
        f.write(f'dur_sec = {dur_sec}\n')
        f.write(f'keypoint_thresh = {keypoint_thresh}\n')
        f.write(f'range_thresh = {range_thresh}\n')
        f.write(f'sd_thresh = {sd_thresh}\n')
        f.write(f'num_of_events_th_low = {num_of_events_th_low}\n')
        f.write(f'num_of_events_th_high = {num_of_events_th_high}\n')
        f.write(f'th_hist = {th_hist}\n')
        f.write(f'plot = {plot}\n')
        f.write(f'save_files = {save_files}\n')
        f.write(f'n_cores = {n_cores}')
    f.close()

    sequences = os.listdir(events_path)
    Parallel(n_jobs=n_cores)(delayed(process_sequence)(
        seq,
        events_path,
        upsamp_frames_path,
        out_path,
        dur_sec,
        num_bins,
        num_event_per_pixel,
        keypoint_thresh,
        range_thresh,
        sd_thresh,
        num_of_events_th_low,
        num_of_events_th_high,
        th_hist,
        plot,
        save_files
    ) for seq in tqdm(sequences))
