import argparse
import os
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename Vimeo90K videos and folders")
    
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of Vimeo90K dataset")

    args = parser.parse_args()
    root_dir = args.root_dir

    # Recursively find all .mp4 files
    mp4_files = glob.glob(os.path.join(root_dir, "**", "*.mp4"), recursive=True)

    # Sort for consistent numbering
    mp4_files.sort()

    for idx, video_path in enumerate(mp4_files, start=1):

        folder_path = os.path.dirname(video_path)

        new_video_name = f"{idx}.mp4"
        new_video_path = os.path.join(folder_path, new_video_name)

        # Rename video first
        os.rename(video_path, new_video_path)

        # Rename folder
        new_folder_path = os.path.join(root_dir, str(idx))
        os.rename(folder_path, new_folder_path)

        print(f"Renamed folder and video to {idx}")

        