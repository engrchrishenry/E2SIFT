# E2SIFT: Neuromorphic SIFT via Direct Feature Pyramid Recovery from Events
This is the official implementation of the IEEE ICIP 2024 paper titled [E2SIFT: Neuromorphic SIFT via Direct Feature Pyramid Recovery from Events  ](https://doi.org/10.1109/ICIP51287.2024.10647465).

<div align="center">
  <img src="figures/overview_e2sift.jpg" alt="Overview E2SIFT" width="590"/>
  <br>
  Figure 1: Overall workflow of the proposed E2SIFT pipeline.
</div>

<br>

<div align="center">
  <img src="figures/overview_keypoint_detection.jpg" alt="Overview Keypoint Detection" width="750"/>
  <br>
  Figure 2: Overall workflow of the proposed alternate keypoint detection.
</div>

## Prerequisites
The code is tested on Linux with the following prerequisites:
1. Python 3.12
2. PyTorch 1.11.0 (CUDA 11.3)
3. MATLAB R2021a
4. VLFeat 0.9.21

Remaining libraries are available in [requirements.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/requirements.txt)

## Installation

- Clone this repository
   ```bash
   git clone https://github.com/engrchrishenry/loc_aware_video_dedup.git
   cd loc_aware_video_dedup
   ```

- Create conda environment
   ```bash
   conda create --name dedup python=3.12
   conda activate dedup
   ```

- Install dependencies
  1. Install [PyTorch](https://pytorch.org/get-started/locally/).
  2. Install [FFmpeg](www.ffmpeg.org/download.html).
  3. The remaining packages can be installed via:
     ```bash
     pip install -r requirements.txt
     ```
  4. For running MATLAB scripts, you are required to install [VLFeat](https://www.vlfeat.org/download.html).
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

## Contact
In case of questions, feel free to contact at chffn@umsystem.edu or engr.chrishenry@gmail.com

