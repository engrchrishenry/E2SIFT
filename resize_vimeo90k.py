import subprocess
import os
import glob

def convert_video(input_file, output_file, target_resolution):
    # Use ffmpeg to resize the video
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f'scale={target_resolution}',
        output_file
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Video resized and saved as {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Video conversion failed.")

def resize_videos_in_folder(input_folder, output_folder, target_resolution):
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv')):
                input_file = os.path.join(root, file)

                # Get the video name without the file extension
                video_name = os.path.splitext(file)[0]

                # Create a subdirectory for each video with the same name as the video
                video_output_folder = os.path.join(output_folder, video_name)
                os.makedirs(video_output_folder, exist_ok=True)

                # Set the output file path inside the subdirectory
                output_file = os.path.join(video_output_folder, file)

                convert_video(input_file, output_file, target_resolution)

if __name__ == "__main__":
    input_folder = '/storage4tb/PycharmProjects/Datasets/Vimeo90K'      # Replace with the path to your input video folder
    output_folder = '/storage4tb/PycharmProjects/Datasets/vimeo_resized'    # Replace with the desired output video folder
    target_resolution = '240:180'     # Set the target resolution (width x height)

    resize_videos_in_folder(input_folder, output_folder, target_resolution)
