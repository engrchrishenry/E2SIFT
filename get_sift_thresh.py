import argparse
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm 


# Function to calculate the number of SIFT keypoints in an image
def calculate_sift_keypoints(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, _ = sift.detectAndCompute(img, None)
    return len(keypoints)


# Function to process images in a directory and create a histogram
def process_images_in_directory(directory_path, num_samples, out_path):
    image_paths = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, filename))

    image_paths = image_paths[:num_samples]

    keypoint_counts = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        num_keypoints = calculate_sift_keypoints(image_path)
        keypoint_counts.append(num_keypoints)

    os.makedirs(out_path, exist_ok=True)

    # Create a histogram of keypoint counts
    plt.hist(keypoint_counts, bins=100)  # Adjust the number of bins as needed
    plt.xlabel('Number of SIFT Keypoints')
    plt.ylabel('Number of Images')
    plt.title('Distribution of SIFT Keypoints')
    plt.savefig(f'{out_path}/sift_hist.png')
    # plt.show()

    print (f"Histogram saved to {out_path}/sift_hist.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a histogram of SIFT keypoints for n number of images in a directory. Useful for setting SIFT keypoint threshold while generating event voxels')
    parser.add_argument('--img_dir', type=str,
                        help='Path to the directory containing images')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to process')
    parser.add_argument('--out_path', type=str, default='output/hists/',
                        help='Path to the output directory')

    args = parser.parse_args()

    img_dir = args.img_dir
    num_samples = args.num_samples
    out_path = args.out_path

    sift = cv2.SIFT_create()
    
    process_images_in_directory(img_dir, num_samples, out_path)
    
