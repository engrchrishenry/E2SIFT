import numpy as np
from PIL import Image, ImageOps
import os
# from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader, myFixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid
import matplotlib.pyplot as plt
import pysift
import cv2
import pandas as pd
import time
import shutil


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
    

if __name__ == "__main__":
    events_path = '/storage4tb/PycharmProjects/rpg_vid2e/example/events_new'
    upsamp_frames_path = '/storage4tb/PycharmProjects/Datasets/reds_120fps_resized_ts'
    # events_path = '/storage4tb/PycharmProjects/rpg_vid2e/example/events_degraded_affine_0.2'
    # upsamp_frames_path = 'output/vimeo_upsampled_degraded/affine/'
    out_path = 'output/ecd_new_ev_th'
    mode = 'fix_dur_sec_events' # fix_dur fix_events fix_dur_sec
    split_name = '' # train valid'
    num_bins = 5
    num_event_per_pixel = 0.55
    dur_sec = 0.25 # in sec
    keypoint_thresh = 50 # None to ignore, 50
    range_thresh = None # None to ignore, 20
    sd_thresh = None # None to ignore, 0.1
    num_of_events_th_low = 100000 # None to ignore, 100000
    num_of_events_th_high = 600000 # None to ignore, 600000
    th_hist = 99.9
    plot = 1
    save_files = 0

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
        f.write(f'keypoint_thresh = {keypoint_thresh}\n')
        f.write(f'num_of_events_th_low = {num_of_events_th_low}\n')
        f.write(f'num_of_events_th_high = {num_of_events_th_high}\n')
        f.write(f'th_hist = {th_hist}\n')
        f.write(f'plot = {plot}\n')
        f.write(f'save_files = {save_files}')
    f.close()

    if mode == 'fix_dur_sec_events':
        sift = cv2.SIFT_create()
        for seq_c, seq in enumerate(os.listdir(events_path)):
            ts_file_frame_path = f'{upsamp_frames_path}/{seq}/timestamps.txt'
            event_files = os.listdir(f'{events_path}/{seq}')
            lines = open(ts_file_frame_path, 'r').readlines()[:len(event_files)]
            lines = [float(temp) for temp in lines]
            if lines != []:
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
                    # print('ts-dur_sec', ts - dur_sec)
                    # print('gt_image_index', gt_im_index)
                    # print('start_frame_index', start_frame_index)
                    # print(ts, lines[gt_im_index])
                    gt_im_name = f'{gt_im_index:08d}.png'  # 000 000 13
                    event_start_name = f'{start_frame_index:010d}.npz'
                    event_last_name = f'{gt_im_index:010d}.npz'
                    # print(seq, event_start_name, event_last_name, gt_im_name)

                    im_name, im_ext = gt_im_name.split('.')

                    im = cv2.imread(f'{upsamp_frames_path}/{seq}/imgs/{gt_im_name}', cv2.IMREAD_GRAYSCALE)

                    # Detect SIFT keypoints and compute descriptors
                    keypoints, descriptors = sift.detectAndCompute(im, None)

                    # image_with_keypoints = cv2.drawKeypoints(im.copy(), keypoints, None)
                    if keypoint_thresh is not None:
                        if len(keypoints) < keypoint_thresh:
                            continue
                            # cv2.imshow('Image with SIFT Keypoints', image_with_keypoints)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                    width = im.shape[1]
                    height = im.shape[0]
                    num_events = int(height * width * num_event_per_pixel)

                    dog = pysift.computeDoGImages(im, num_intervals=2)
                    x, y, t, p = [], [], [], []
                    for event_name in range(int(os.path.splitext(event_start_name)[0]), int(os.path.splitext(event_last_name)[0])+1):
                        event_name = f'{event_name:010d}.npz'
                        xx = np.load(f'{events_path}/{seq}/{event_name}')
                        x.append(xx['x'])
                        y.append(xx['y'])
                        t.append(xx['t'].astype(np.float32)/1e+9)
                        p.append(xx['p'])

                    x = np.concatenate(x, axis=0)
                    y = np.concatenate(y, axis=0)
                    t = np.concatenate(t, axis=0)
                    p = np.concatenate(p, axis=0)

                    if num_of_events_th_low is not None or num_of_events_th_high is not None:
                        # if len(t) < num_events-3000 or len(t) > num_events+5000:
                        #     continue
                        if len(t) < num_of_events_th_low or len(t) > num_of_events_th_high:
                            continue

                    event_df = pd.DataFrame({'t': t, 'x': x, 'y': y, 'p': p})
                    event_window = np.array(event_df.values)

                    vox = events_to_voxel_grid(event_window, num_bins=num_bins,
                                            width=width,
                                            height=height)
                    
                    range_vox = vox.max() - vox.min()
                    avg_sd = 0
                    sd_indicator = 0
                    for temp in vox:
                        sd_temp = np.std(temp)
                        if sd_thresh is not None:
                            if sd_temp < sd_thresh:
                                sd_indicator = 1
                                break
                        avg_sd += sd_temp
                    avg_sd = avg_sd/len(vox)
                    if range_thresh is not None:
                        if range_vox > range_thresh:
                            continue
                    if sd_thresh is not None:
                        if sd_indicator == 1:
                            continue
                    
                    stats = {
                        "range_vox": float(range_vox),
                        "vox_max": float(vox.max()),
                        "vox_min": float(vox.min()),
                        "sds": list(np.std(v) for v in vox),
                        "avg_sd": float(avg_sd),
                        "num_events": float(len(x))
                    }

                    if save_files == 1:
                        # np.save(f'{out_path}/vox/{seq}_{im_name}.npy', vox)
                        # np.save(f'{out_path}/dog/{seq}_{im_name}.npy', dog)
                        np.savez_compressed(f'{out_path}/vox/{seq}_{im_name}', vox)
                        np.savez_compressed(f'{out_path}/dog/{seq}_{im_name}', dog)
                        np.savez_compressed(f'{out_path}/stats/{seq}_{im_name}', stats=stats)
                        shutil.copy(f'{upsamp_frames_path}/{seq}/imgs/{im_name}.{im_ext}',
                                f'{out_path}/images/{seq}_{im_name}.{im_ext}')
                    
                    if plot == 1:
                        fig, axes = plt.subplots(4, len(vox), figsize=(13, 9))  # 1 row, 2 columns
                        plt.suptitle(f'{len(x)}/{num_events} range {range_vox:.3f} max {vox.max():.3f} min {vox.min():.3f}, avg_sd {avg_sd:.3f}, clip_thresh {th_hist}')
                        
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

            seq_c += 1
            print(seq, seq_c, '/', len(os.listdir(events_path)))


    if mode == 'fix_dur':
        seq_c = 1
        for seq in os.listdir(events_path):
            ts_file_frame_path = f'{upsamp_frames_path}/{seq}/timestamps.txt'
            durations_s, _ = get_duration_esim(ts_file_frame_path, f'{events_path}/{seq}/', start_frame, num_of_frames)
            c = 0
            for temp in durations_s:
                duration_s, im_name, list_of_event_files = temp
                im_name, im_ext = im_name.split('.')
                im = cv2.imread(f'{upsamp_frames_path}/{seq}/imgs/{im_name}.{im_ext}', cv2.IMREAD_GRAYSCALE)

                width = im.shape[1]
                height = im.shape[0]

                dog = pysift.computeDoGImages(im, num_intervals=2)
                x, y, t, p = [], [], [], []
                for event in list_of_event_files:
                    xx = np.load(f'{events_path}/{seq}/{event}')
                    x.append(xx['x'])
                    y.append(xx['y'])
                    t.append(xx['t'].astype(np.float32)/1e+6)
                    p.append(xx['p'])

                x = np.concatenate(x, axis=0)
                y = np.concatenate(y, axis=0)
                t = np.concatenate(t, axis=0)
                p = np.concatenate(p, axis=0)

                event_df = pd.DataFrame({'t': t, 'x': x, 'y': y, 'p': p})
                event_window = np.array(event_df.values)

                vox = events_to_voxel_grid(event_window, num_bins=num_bins,
                                                                    width=width,
                                                                    height=height)
                np.save(f'{out_path}/vox/{seq}_{im_name}.npy', vox)
                np.save(f'{out_path}/dog/{seq}_{im_name}.npy', dog)
                shutil.copy(f'{upsamp_frames_path}/{seq}/imgs/{im_name}.{im_ext}', f'{out_path}/images/{seq}_{im_name}.{im_ext}')
                if c % 10 == 0:
                    print (seq, seq_c, '/', len(os.listdir(events_path)), c, '/', len(durations_s))
                c += 1
                if plot == 1:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(vox[4], cmap='gray')
                    axes[0].set_title('Vox')
                    axes[1].imshow(dog[0], cmap='gray')
                    axes[1].set_title('DoG')
                    plt.tight_layout()
                    plt.savefig(f'{out_path}/plots/{seq}_{im_name}.{im_ext}')
                    plt.close()
            seq_c += 1

    if mode == 'fix_events':
        fps_list = []
        seq_c = 1
        for seq in os.listdir(events_path):
            ts_file_frame_path = f'{upsamp_frames_path}/{seq}/timestamps.txt'
            frames = os.listdir(f'{upsamp_frames_path}/{seq}/imgs')

            f = open(ts_file_frame_path, 'r')
            lines = f.readlines() # [:len(os.listdir(events_path))]
            lines = [x.replace('\n', '') for x in lines]
            # print ((float(lines[4]) - float(lines[2]))*1000)
            fps = len(lines)/(float(lines[-1]) - float(lines[0]))
            fps_list.append(fps)
            # print (float(lines[-1]), len(lines))
            print ('seq', seq, 'fps', fps)
            # exit(0)


            events_arr = []
            x, y, t, p = [], [], [], []
            for event in os.listdir(f'{events_path}/{seq}'):
                event_int = int(event.replace('.npz', ''))
                for xx in frames:
                    if int(xx.replace('.png', '')) == event_int:
                        im_name = xx
                        continue
                im = cv2.imread(f'{upsamp_frames_path}/{seq}/imgs/{im_name}', cv2.IMREAD_GRAYSCALE)

                width = im.shape[1]
                height = im.shape[0]

                num_events = int(height * width * num_event_per_pixel)
                temp = np.load(f'{events_path}/{seq}/{event}')
                x.append(temp['x'])
                y.append(temp['y'])
                # print (temp['t'])
                # exit(0)
                # t.append(temp['t'].astype(np.float32)/1e+6)
                t.append(temp['t'].astype(np.float32))

                p.append(temp['p'])
            x = np.concatenate(x)
            y = np.concatentate(y)
            t = np.concatenate(t)
            p = np.concatenate(p)

            event_df = pd.DataFrame({'t': t, 'x': x, 'y': y, 'p': p})
            print ('full dataframe shape', event_df.shape)
            print ('num_events', num_events)
            num_chunks = len(event_df) // num_events
            chunks = [event_df.iloc[i * num_events: (i + 1) * num_events] for i in range(num_chunks)]

            for chunk in chunks:
                # print (chunk)
                print ('len(chunk)', len(chunk))
                event_window = np.array(chunk.values)
                print (event_window[-1][0], event_window[0][0], ( event_window[-1][0] - event_window[0][0] ))

                vox = events_to_voxel_grid(event_window, num_bins=num_bins,
                                                                width=width,
                                                                height=height)
                exit(0)
            #
            #
            #
            #
            #
            #
