import subprocess
import os
import multiprocessing

def convert_video(input_file, output_file, target_resolution, desired_duration=None):
    # Use ffmpeg to resize the video
    # cmd = [
    #     'ffmpeg',
    #     '-i', input_file,
    #     '-vf', f'scale={target_resolution}',
    #     output_file
    # ]

    if desired_duration is None:
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ss', '5',
            '-vf', f'scale={target_resolution}',
            '-an',                           # Disable audio output
            output_file
        ]
    else:
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ss', '5',
            '-t', str(desired_duration),
            '-vf', f'scale={target_resolution}',
            '-an',                           # Disable audio output
            output_file
        ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Video resized and saved as {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Video conversion failed.")

def worker(input_file, output_folder, target_resolution, desired_duration):
    # Get the video name without the file extension
    video_name = os.path.splitext(os.path.basename(input_file))[0]

    # Create a subdirectory for each video with the same name as the video
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    # Set the output file path inside the subdirectory
    output_file = os.path.join(video_output_folder, os.path.basename(input_file))

    convert_video(input_file, output_file, target_resolution, desired_duration)

def resize_videos_in_folder(input_folder, output_folder, target_resolution, desired_duration):
    os.makedirs(output_folder, exist_ok=True)

    video_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv')):
                input_file = os.path.join(root, file)
                video_files.append(input_file)

    # Create a pool of worker processes
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # Resize videos in parallel
    pool.starmap(worker, [(video, output_folder, target_resolution, desired_duration) for video in video_files])

    pool.close()
    pool.join()

if __name__ == "__main__":
    input_folder = '/storage4tb/PycharmProjects/Datasets/Vimeo90K'      # Replace with the path to your input video folder
    output_folder = '/storage4tb/PycharmProjects/Datasets/vimeo_resized'    # Replace with the desired output video folder
    target_resolution = '240:180' # Set the target resolution (width x height) '240:180'
    desired_duration = None # Either 'None' or the number of seconds e.g. 1.

    resize_videos_in_folder(input_folder, output_folder, target_resolution, desired_duration)


'''
python esim_torch/scripts/generate_events.py --input_dir=/storage4tb/PycharmProjects/Datasets/reds_120fps_ts/ \
                                     --output_dir=example/events_new \
                                     --contrast_threshold_neg=0.2 \
                                     --contrast_threshold_pos=0.2 \
                                     --refractory_period_ns=0

python upsampling/upsample.py --input_dir=/storage4tb/PycharmProjects/Datasets/vimeo_resized --output_dir=/storage4tb/PycharmProjects/Datasets/vimeo_upsampled/
'''
