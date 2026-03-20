import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a histogram for event voxels or LoG pyramids. Useful for setting clipping threshold for event voxels or LoG pyramids during training TSFNet_E2SIFT')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the directory containing events voxels or log pyramids')
    parser.add_argument('--save_file', type=str, required=True,
                        help='Path to the output file (including filename with extension, e.g., output/hists/vox_hist.png)')
    parser.add_argument('--samp_percent', type=int, default=5,
                        help='Percentage of pixels to sample')
    parser.add_argument('--hist_pctl', type=float, default=99.9,
                        help='To mark where --hist_pctl percentile of the data lies in the histogram')

    args = parser.parse_args()

    data_path = args.data_path
    save_file = args.save_file
    sample_percent = args.samp_percent
    hist_pctl = args.hist_pctl

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)

    all_samples = []
    for i, fname in enumerate(sorted(os.listdir(data_path))):
        if fname.endswith(".npz"):
            arr = np.load(os.path.join(data_path, fname))['arr_0']
        if fname.endswith(".mat"):
            arr = scipy.io.loadmat(os.path.join(data_path, fname))['log_pyramid']
            
        arr = arr.reshape(-1)

        total_pixels = arr.size
        num_samples = int((sample_percent / 100) * total_pixels)
        sampled = np.random.choice(arr.flatten(), size=num_samples, replace=False)
        all_samples.append(sampled)
        
        if i % 100 == 0:
            print (f'Processed {i}/{len(os.listdir(data_path))}')

    # Combine and flatten all sampled data
    all_samples = np.concatenate(all_samples)

    # Compute percentiles
    lower = np.percentile(all_samples, 100-hist_pctl)
    upper = np.percentile(all_samples, hist_pctl)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_samples, bins=100, color='skyblue', edgecolor='black')
    plt.axvline(lower, color='red', linestyle='--', label=f'{100-hist_pctl:.2f} percentile: {lower:.2f}')
    plt.axvline(upper, color='green', linestyle='--', label=f'{hist_pctl:.2f} percentile: {upper:.2f}')
    plt.title("Histogram of Sampled Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_file)
    print(f'Histogram saved to {save_file}')

