# E2SIFT: Neuromorphic SIFT via Direct Feature Pyramid Recovery from Events
This is the official implementation of the IEEE ICIP 2024 paper titled [E2SIFT: Neuromorphic SIFT via Direct Feature Pyramid Recovery from Events  ](https://doi.org/10.1109/ICIP51287.2024.10647465).

<br>

<div align="center">
  <img src="figures/overview_e2sift.jpg" alt="Overview E2SIFT" width="590"/>
  <br>
  Overall workflow of the proposed E2SIFT pipeline.
</div>

<br>

<div align="center">
  <img src="figures/overview_keypoint_detection.jpg" alt="Overview Keypoint Detection" width="750"/>
  <br>
  Overall workflow of the proposed alternate keypoint detection.
</div>

## ⚠️Prerequisites
The code is tested on Linux with the following prerequisites:
1. Python 3.12
2. PyTorch 1.11.0 (CUDA 11.3)
3. MATLAB R2021a
4. VLFeat 0.9.21
5. pip install yt-dlp 
pip install -U yt-dlp secretstorage
Remaining libraries are available in [requirements.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/requirements.txt)

## Installation

- Clone this repository
   ```bash
   git clone https://github.com/engrchrishenry/E2SIFT.git
   cd E2SIFT
   ```

- Create conda environment
   ```bash
   conda create --name e2sift python=3.13
   conda activate e2sift
   ```

- Install dependencies
  1. Install [PyTorch](https://pytorch.org/get-started/locally/).
  2. Install [FFmpeg](www.ffmpeg.org/download.html).
  3. The remaining packages can be installed via:
     ```bash
     pip install -r requirements.txt
     ```

## Dataset Preparation

<!-- ### Option 1: Use Pre-computed Data -->

<!--Download -->

<!--### Download Dataset-->
The E2SIFT paper used a subset from the [Event Camera Dataset](https://rpg.ifi.uzh.ch/davis_data.html) (real events) and [Vimeo-90k Dataset](http://toflow.csail.mit.edu) ([ESIM](https://github.com/uzh-rpg/rpg_vid2e/tree/master)-generated synthetic events) for training and testing. A subset from the [Event Camera Dataset](https://rpg.ifi.uzh.ch/davis_data.html) (real events) was used for testing.
  
### [Event Camera Dataset](https://rpg.ifi.uzh.ch/davis_data.html)

- Download sequences:
  
  We provide links to the train and test sequences used in the E2SIFT paper. The 'office_zigzag.zip' sequence was excluded as explained in the paper.

  Download and the train/test sequences as mentioned in the E2SIFT paper.

  ```bash
  # Download train sequences
  wget -i data/ecd_train_links.txt -P ecd/train

  # Download test sequences
  wget -i data/ecd_test_links.txt -P ecd/test
  ```

  Unzip train/test sequences

  ```bash
  # Unzip train sequences
  for f in ecd/train/*.zip; do unzip -o "$f" -d "${f%.zip}"; done
  
  # Unzip test sequences
  for f in ecd/test/*.zip; do unzip -o "$f" -d "${f%.zip}"; done
  ```

  [Optional] Remove the .zip files to save storage space.
  ```bash
  # Delete train sequences
  rm ecd/train/*.zip

  # Delete test sequences
  rm ecd/test/*.zip
  ```

- Event voxel generation

  ```bash
  # Generate event voxels for training the model
  python prep_data_ecd_multi_core.py --events_dir ecd/train/ --out_dir <output_path>

  # Generate event voxels for testing the model
  python prep_data_ecd_multi_core.py --events_dir ecd/test/ --out_dir <output_path>
  ```

  Usage:

  ```bash
  options:
    -h, --help            show this help message and exit
    --events_dir EVENTS_DIR
                          Path to directory containing ESIM-generated synthetic events
    --out_dir OUT_DIR     Path to output directory
    --bins BINS           Number of bins for event voxel generation
    --dur_sec DUR_SEC     Event window duration in seconds
    --res RES             Event camera resolution (e.g., '240:180')
    --events_per_px EVENTS_PER_PX
                          Number of events per pixel
    --kp_th KP_TH         Keypoint threshold for rejecting blank frames. None to ignore.
    --sd_th SD_TH         Standard deviation threshold. None to ignore.
    --range_th RANGE_TH   Range (max value - min value) threshold. None to ignore.
    --th_hist TH_HIST     Clipping threshold for histogram plotting. Value between 0 and 100, e.g., 99.9 means clipping at 99.9 percentile.
    --plot PLOT           Plot figures. True -> save plots: False -> do not save plots
    --cores CORES         Number of cores to use. -1 -> use all cores.
  ```

  - Generate Laplacian of Gaussian (LoG) (MATLAB)

  Update `images_dir`, `output_dir`, and `n_cores` variables in MATLAB script [data_processing/LoG_generation/gen_log_data.m](https://github.com/engrchrishenry/E2SIFT/blob/main/data_processing/LoG_generation/gen_LoG_data.m).
  
  Run [data_processing/LoG_generation/gen_log_data.m](https://github.com/engrchrishenry/E2SIFT/blob/main/data_processing/LoG_generation/gen_LoG_data.m) to generate the LoG data.

### [Vimeo-90k Dataset](http://toflow.csail.mit.edu)

  - Download videos:

    The list of vimeo video links is available [here](https://data.csail.mit.edu/tofu/dataset/original_video_list.txt). We provide a helper script to batch download the videos.
    ```bash
    cd data_processing/
    python download_vimeo90k.py --video_links data/original_video_list.txt -- out_path <ouptut_directory> --cores 2
    ```
    Use less `--cores` to avoid "HTTP Error 429: Too Many Requests".

    `download_vimeo90k.py` downloads the lowest quality video available   (without audio). Modify the `ydl_opts` in `download_vimeo90k.py` to change this behavior.

  - Rename video files and folders (important for synthetic event generation via [ESIM](https://github.com/uzh-rpg/rpg_vid2e/tree/master)). The [generate_events.py](https://github.com/uzh-rpg/rpg_vid2e/blob/master/esim_torch/scripts/generate_events.py) script from [ESIM](https://github.com/uzh-rpg/rpg_vid2e/tree/master) will fail without renaming due to the presence special characters in video filenames.
    ```bash
    python rename_vimeo90k.py --root_dir <vimeo90k_dataset_path>
    ```

  - Resize videos (important for synthetic event generation via [ESIM](https://github.com/uzh-rpg/rpg_vid2e/tree/master)).
    ```bash
    python resize_vimeo90k_multi_core.py --video_dir <vimeo90k_dataset_path> --out_path <ouptut_directory> --res <width:height> --cores -1
    ```
    Use `240:180` for `<width:height>` if you want to be consistent with the paper.

- Synthesize events
  
  Follow the instructions [here](https://github.com/uzh-rpg/rpg_vid2e/tree/master) to setup ESIM and build the python binding with GPU support. Use a different conda environment with the exact versions of the dependencies reqired to run ESIM with GPU support. Once ESIM is setup:

  Upsample Vimeo-90k videos to a higher FPS via [upsample.py](https://github.com/uzh-rpg/rpg_vid2e/blob/master/upsampling/upsample.py). Sample command:
  ```bash
  python upsampling/upsample.py --input_dir=<resized_videos_path> --output_dir=<upsampled_output_path>
  ```

  Generate synthetic events via [generate_events.py](https://github.com/uzh-rpg/rpg_vid2e/blob/master/esim_torch/scripts/generate_events.py). Sample command:

  ```bash
  python esim_torch/scripts/generate_events.py --input_dir=<upsampled_videos_path> \
    --output_dir=<events_output_path> \
    --contrast_threshold_neg=0.2 \
    --contrast_threshold_pos=0.2 \
    --refractory_period_ns=0
  ```
  
- Event voxel generation

  Generate event voxels for training the model.

  ```bash
  python prep_data_esim_multi_core.py --events_dir <synthetic_events_path> \
    --upsamp_frames_dir <upsampled_frames_path> \
    --out_dir <output_path>
  ```
  
  Usage:

  ```bash
  options:
    -h, --help            show this help message and exit
    --events_dir EVENTS_DIR
                          Path to directory containing ESIM-generated synthetic events
    --upsamp_frames_dir UPSAMP_FRAMES_DIR
                          Path to directory containing upsampled frames
    --out_dir OUT_DIR     Path to output directory
    --bins BINS           Number of bins for event voxel generation
    --dur_sec DUR_SEC     Event window duration in seconds
    --res RES             Event camera resolution (e.g., '240:180')
    --events_per_px EVENTS_PER_PX
                          Number of events per pixel
    --kp_th KP_TH         Keypoint threshold for rejecting blank frames. None to ignore.
    --sd_th SD_TH         Standard deviation threshold. None to ignore.
    --range_th RANGE_TH   Range (max value - min value) threshold. None to ignore.
    --th_hist TH_HIST     Clipping threshold for histogram plotting. Value between 0 and 100, e.g., 99.9 means clipping at 99.9 percentile.
    --plot PLOT           Plot figures. True -> save plots: False -> do not save plots
    --cores CORES         Number of cores to use. -1 -> use all cores.
  ```

- Generate Laplacian of Gaussian (LoG) (MATLAB)

  Update `images_dir`, `output_dir`, and `n_cores` variables in MATLAB script [data_processing/LoG_generation/gen_log_data.m](). Set `n_cores` to maximum number of CPU cores available for fastest processing.
  
  Run [data_processing/LoG_generation/gen_log_data.m]() to generate the LoG data.
  

  


## Citation

If you use this work, please cite:

```bibtex
@INPROCEEDINGS{10647465,
  author={Henry, Chris and Maharjan, Paras and Li, Zhu and York, George},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
  title={E2SIFT: Neuromorphic SIFT via Direct Feature Pyramid Recovery from Events}, 
  year={2024},
  volume={},
  number={},
  pages={2786-2792},
  keywords={Thresholding (Imaging);Neuromorphics;Detectors;Transforms;Streaming media;Vision sensors;Cameras;neuromorphic vision sensor;event camera;scale-invariant feature transform;SIFT;keypoint detection},
  doi={10.1109/ICIP51287.2024.10647465}}

```

<mark>⚠️You may also explore our work on event-based object detection [here](). Consider citing the following:</mark>

<!--
```bibtex
@INPROCEEDINGS{10095417,
  author={Henry, Chris and Liao, Rijun and Lin, Ruiyuan and Zhang, Zhebin and Sun, Hongyu and Li, Zhu},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Lightweight Fisher Vector Transfer Learning for Video Deduplication}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Computational modeling;Transfer learning;Transforms;Multilayer perceptrons;Signal processing;Robustness;Encoding;Video deduplication;near-duplicate video detection;near-duplicate video copy detection;fisher vector aggregation},
  doi={10.1109/ICASSP49357.2023.10095417}}
```
-->

## Acknowledgements

This work was supported by the National Science Foundation (NSF) under Award 2148382.

- [TSF-Net](ieeexplore.ieee.org/document/10275101/) model implementation were adapted from: https://github.com/parasmaharjan/TSFNet  
- Some parts of the event data processing were adapted from: https://github.com/uzh-rpg/rpg_e2vid

We gratefully acknowledge the authors and contributors for making their work publicly available.

## Contact
In case of questions, feel free to contact at chffn@umsystem.edu or engr.chrishenry@gmail.com

