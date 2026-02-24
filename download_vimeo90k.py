import argparse
import yt_dlp
from yt_dlp.utils import DownloadError
import os
import multiprocessing


# Define the function to download videos from links in a text file
def download_video(url, output_directory):
    ydl_opts = {
        'format': 'worstvideo', # worst best
        'outtmpl': f'{output_directory}/%(title)s/%(title)s.%(ext)s',
        'cookiesfrombrowser': ('chrome',),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
        except DownloadError as e:
            print(f"Download error for URL: {url}\nError: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Video Deduplication and Localization With Temporal Consistence Re-Ranking")

    parser.add_argument("--video_links", type=str, required=True,
                        help="Path to text file containng video links for Vimeo90K dataset")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path to output folder")
    
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    # Read Vimeo links from the text file
    with open(args.video_links, 'r') as file:
        vimeo_links = [line.strip() for line in file]

    # Create a pool of worker processes
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # Download videos in parallel
    results = [pool.apply_async(download_video, args=(link, args.out_path)) for link in vimeo_links]

    # Wait for all processes to finish
    pool.close()
    pool.join()

    for result in results:
        result.get()  # Retrieve results (errors if any)

    print("All downloads completed.")

