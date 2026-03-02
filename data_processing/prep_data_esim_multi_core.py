import argparse
import numpy as np
from PIL import Image, ImageOps
import os
from utils.inference_utils import events_to_voxel_grid
import cv2
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


def process_sequence(seq, events_dir, upsamp_frames_dir, out_dir, dur_sec, num_bins, num_event_per_pixel,
                     kp_th, range_thresh, sd_thresh, events_th_low, events_th_high, th_hist, plot, save_files):
    sift = cv2.SIFT_create()
    ts_file_frame_path = f'{upsamp_frames_dir}/{seq}/timestamps.txt'
    event_files = os.listdir(f'{events_dir}/{seq}')
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
            im = cv2.imread(f'{upsamp_frames_dir}/{seq}/imgs/{gt_im_name}', cv2.IMREAD_GRAYSCALE)
        except:
            continue

        if im is None:
            continue

        keypoints, descriptors = sift.detectAndCompute(im, None)
        if kp_th is not None and len(keypoints) < kp_th:
            continue

        width, height = im.shape[1], im.shape[0]
        num_events = int(height * width * num_event_per_pixel)

        # Aggregate events
        x, y, t, p = [], [], [], []
        for event_name in range(start_frame_index, gt_im_index + 1):
            ev_file = f'{event_name:010d}.npz'
            ev_path = f'{events_dir}/{seq}/{ev_file}'
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

        if ((events_th_low is not None and len(t) < events_th_low) or
            (events_th_high is not None and len(t) > events_th_high)):
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
            np.savez_compressed(f'{out_dir}/vox/{seq}_{im_name}', vox=vox)
            np.savez_compressed(f'{out_dir}/stats/{seq}_{im_name}', stats=stats)
            shutil.copy(f'{upsamp_frames_dir}/{seq}/imgs/{gt_im_name}',
                        f'{out_dir}/images/{seq}_{im_name}.{im_ext}')
        
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
            plt.savefig(f'{out_dir}/plots/{seq}_{im_name}.{im_ext}')
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate event voxels from ESIM-generated synthetic events.")

    parser.add_argument("--events_dir", type=str, required=True,
                        help="Path to directory containing ESIM-generated synthetic events")
    parser.add_argument('--upsamp_frames_dir', type=str, required=True,
                        help='Path to directory containing upsampled frames')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--bins', type=int, default=5,
                        help='Number of bins for event voxel generation)')
    parser.add_argument('--events_per_px', type=float, default=0.55,
                        help='Number of events per pixel')
    parser.add_argument('--dur_sec', type=float, default=0.05,
                        help='Event window duration')
    parser.add_argument('--kp_th', type=int, default=50,
                        help='Keypoint threshold for rejecting blank frames. None to ignore.')
    parser.add_argument('--sd_th', type=float, default=0.1,
                        help='Standard deviation threshold. None to ignore.')
    parser.add_argument('--range_th', type=int, default=20,
                        help='Range (max value - min value) threshold. None to ignore.')
    parser.add_argument('--events_th_low', type=int, default=70000,
                        help='Minimum number of events within an event windows. None to ignore.')
    parser.add_argument('--events_th_high', type=int, default=300000,
                        help='Maximum number of events within an event windows. None to ignore.')
    parser.add_argument('--th_hist', type=float, default=100,
                        help='Clipping threshold for histogram plotting. Value between 0 and 100, e.g., 99.9 means clipping at 99.9 percentile.')
    parser.add_argument('--plot', type=bool, default=True,
                        help='Plot figures. True -> save plots: False -> do not save plots')
    parser.add_argument('--save_files', type=int, default=1,
                        help='Save output data. True -> save output files; False -> do not save output files')
    parser.add_argument('--cores', type=int, default=-1,
                        help='Number of cores to use. -1 -> use all cores.')

    args = parser.parse_args()
    events_dir = args.events_dir
    upsamp_frames_dir = args.events_dir
    out_dir = args.out_dir
    bins = args.bins
    events_per_px = args.events_per_px
    dur_sec = args.dur_sec
    kp_th = args.kp_th
    range_th = args.range_th
    sd_th = args.sd_th
    events_th_low = args.events_th_low
    events_th_high = args.events_th_high
    th_hist = args.th_hist
    plot = args.plot
    save_files = args.save_files
    cores = args.cores

    out_name = f'{bins}_{events_per_px}_{dur_sec}_{kp_th}_{events_th_low}_{events_th_high}'
    out_dir = f'{out_dir}/{out_name}'

    for sub in ["images", "vox", "plots", "stats"]:
        os.makedirs(f"{out_dir}/{sub}", exist_ok=True)
    
    with open(f"{out_dir}/params.txt", "w") as f:
        f.write(f'events_dir = {events_dir}\n')
        f.write(f'upsamp_frames_dir = {upsamp_frames_dir}\n')
        f.write(f'out_dir = {out_dir}\n')
        f.write(f'num_bins = {bins}\n')
        f.write(f'events_per_px = {events_per_px}\n')
        f.write(f'dur_sec = {dur_sec}\n')
        f.write(f'kp_th = {kp_th}\n')
        f.write(f'range_thresh = {range_th}\n')
        f.write(f'sd_thresh = {sd_th}\n')
        f.write(f'events_th_low = {events_th_low}\n')
        f.write(f'events_th_high = {events_th_high}\n')
        f.write(f'th_hist = {th_hist}\n')
        f.write(f'plot = {plot}\n')
        f.write(f'save_files = {save_files}\n')
        f.write(f'cores = {cores}')
    f.close()

    sequences = os.listdir(events_dir)
    Parallel(n_jobs=n_cores)(delayed(process_sequence)(
        seq,
        events_dir,
        upsamp_frames_dir,
        out_dir,
        dur_sec,
        num_bins,
        events_per_px,
        kp_th,
        range_thresh,
        sd_thresh,
        events_th_low,
        events_th_high,
        th_hist,
        plot,
        save_files
    ) for seq in tqdm(sequences))

