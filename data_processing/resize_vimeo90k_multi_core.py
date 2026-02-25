import argparse
import subprocess
import os
import multiprocessing

def convert_video(input_file, output_file, target_resolution, desired_duration=None):
    if desired_duration is None:
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ss', '5',                             # Start processing from 5 seconds
            '-vf', f'scale={target_resolution}',    # Resize video
            '-an',                                  # Remove audio stream from output
            output_file
        ]
    else:
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ss', '5',                             # Start processing from 5 seconds
            '-t', str(desired_duration),            # Output duration in seconds
            '-vf', f'scale={target_resolution}',    # Resize video
            '-an',                                  # Remove audio stream from output
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

def resize_videos_in_folder(input_folder, output_folder, target_resolution, desired_duration, cores):
    os.makedirs(output_folder, exist_ok=True)

    video_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv')):
                input_file = os.path.join(root, file)
                video_files.append(input_file)

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=cores)

    # Resize videos in parallel
    pool.starmap(worker, [(video, output_folder, target_resolution, desired_duration) for video in video_files])

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Vimeo-90k Dataset videos")

    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to input video folder for Vimeo90K dataset")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path to output folder")
    parser.add_argument("--res", type=str, required=True,
                        help="Target resolution (e.g., '240:180')")
    parser.add_argument("--cores", type=int, default=-1,
                        help="Number of cores to use to process the data. Default: -1 -> Uses all cores.")
    parser.add_argument("--dur", type=int, default=None,
                        help="Target duration in seconds (e.g., '10'). Set to None to keep original duration.")
                        
    args = parser.parse_args()

    input_folder = args.video_dir
    output_folder = args.out_path
    target_resolution = args.res # Set the target resolution (width x height) '240:180'
    desired_duration = args.dur if args.dur is not None else None
    cores = multiprocessing.cpu_count() if args.cores == -1 else args.cores

    resize_videos_in_folder(input_folder, output_folder, target_resolution, desired_duration, cores)

