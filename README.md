# Markerless Kinematic Analysis (MKA)

[![MKA-Demo]](https://github.com/user-attachments/assets/20d603a7-c2c9-42ed-ba3f-a96d68d19b4b)


##  Setup Instructions

### 1. Environment Setup
Create and configure the conda environment:

```bash
conda create -n mka python=3.8 -y
conda activate mka
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py38_cu117_pyt1131.tar.bz2
pip install -r requirements.txt

cd dependencies
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ..
pip install -v -e dependencies/ViTPose

cd dependencies/cpp_module
sh install.sh 
cd ../..
```

If build cpp_module fail, you can try conda-based gcc:
```bash
conda install -c conda-forge gcc=9 gxx=9
conda install -c conda-forge libxcrypt
```

You also need to setup SAM2 environment following [facebookresearch/sam2](https://github.com/facebookresearch/sam2)

### 2. Human Models
Human body model files are required for body, hand, and face parameterization.

```
human_models/
│── human_models.py
└── human_model_files/
    ├── J_regressor_extra.npy
    ├── J_regressor_h36m.npy
    ├── mano_mean_params.npz
    ├── smpl_mean_params.npz
    ├── smpl/
    │   ├── SMPL_FEMALE.pkl
    │   ├── SMPL_MALE.pkl
    │   └── SMPL_NEUTRAL.pkl
    ├── smplx/
    │   ├── SMPLX_FEMALE.pkl
    │   ├── SMPLX_MALE.pkl
    │   ├── SMPLX_NEUTRAL.pkl
    │   ├── SMPLX_to_J14.pkl
    │   ├── SMPL-X__FLAME_vertex_ids.npy
    │   └── MANO_SMPLX_vertex_ids.pkl
    └── mano/
        └── MANO_RIGHT.pkl
```

Here we provide some download links for the files:
- [SMPL](https://smpl-x.is.tue.mpg.de/)
- [SMPLX](https://smpl-x.is.tue.mpg.de/)
- [MANO](https://mano.is.tue.mpg.de/)
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)


### 3. Pretrained Weights
Pretrained models are required for pose detection and human mesh recovery.

```
pretrained_models/
├── yolov8x.pt
├── sam2.1_hiera_large.pt
├── hamer_ckpts/
│   ├── dataset_config.yaml
│   ├── model_config.yaml
│   └── checkpoints/
│       └── hamer.ckpt
├── smplest_x_h/
│   ├── config_base.py
│   └── smplest_x_h.pth.tar
└── vitpose_ckpts/
    └── vitpose+_huge/
        └── wholebody.pth
```

Some instructions:
 - [smplest_x_h and yolov8x.pt](https://github.com/SMPLCap/SMPLest-X?tab=readme-ov-file#preparation)
 - [hamer_ckpts and vitpose_ckpts](https://github.com/geopavlakos/hamer/blob/main/fetch_demo_data.sh)
 - [sam2.1](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints)

##  Running the Pipeline

To execute the full processing pipeline:
```bash
bash run_pipeline.sh
```

## Explore More [Motrix](https://github.com/MotrixLab) Projects

### Motion Capture
- [SMPL-X] [TPAMI'25] [SMPLest-X](https://github.com/MotrixLab/SMPLest-X): An extended version of [SMPLer-X](https://github.com/MotrixLab/SMPLer-X) with stronger foundation models.
- [SMPL-X] [NeurIPS'23] [SMPLer-X](https://github.com/MotrixLab/SMPLer-X): Scaling up EHPS towards a family of generalist foundation models.
- [SMPL-X] [ECCV'24] [WHAC](https://github.com/MotrixLab/WHAC): World-grounded human pose and camera estimation from monocular videos.
- [SMPL-X] [CVPR'24] [AiOS](https://github.com/MotrixLab/AiOS): An all-in-one-stage pipeline combining detection and 3D human reconstruction. 
- [SMPL-X] [NeurIPS'23] [RoboSMPLX](https://github.com/MotrixLab/RoboSMPLX): A framework to enhance the robustness of whole-body pose and shape estimation.
- [SMPL-X] [ICML'25] [ADHMR](https://github.com/MotrixLab/ADHMR): A framework to align diffusion-based human mesh recovery methods via direct preference optimization.
- [SMPL-X] [MKA](https://github.com/MotrixLab/MKA): Full-body 3D mesh reconstruction from single- or multi-view RGB videos.
- [SMPL] [ICCV'23] [Zolly](https://github.com/MotrixLab/Zolly): 3D human mesh reconstruction from perspective-distorted images.
- [SMPL] [IJCV'26] [PointHPS](https://github.com/MotrixLab/PointHPS): 3D HPS from point clouds captured in real-world settings.
- [SMPL] [NeurIPS'22] [HMR-Benchmarks](https://github.com/MotrixLab/hmr-benchmarks): A comprehensive benchmark of HPS datasets, backbones, and training strategies.

### Motion Generation
- [SMPL-X] [ICLR'26] [ViMoGen](https://github.com/MotrixLab/ViMoGen): A comprehensive framework that transfers knowledge from ViGen to MoGen across data, modeling, and evaluation.
- [SMPL-X] [ECCV'24] [LMM](https://github.com/MotrixLab/LMM): Large Motion Model for Unified Multi-Modal Motion Generation.
- [SMPL-X] [NeurIPS'23] [FineMoGen](https://github.com/MotrixLab/FineMoGen): Fine-Grained Spatio-Temporal Motion Generation and Editing.
- [SMPL] [InfiniteDance](https://github.com/MotrixLab/InfiniteDance): A large-scale 3D dance dataset and an MLLM-based music-to-dance model designed for robust in-the-wild generalization.
- [SMPL] [NeurIPS'23] [InsActor](https://github.com/MotrixLab/insactor): Generating physics-based human motions from language and waypoint conditions via diffusion policies.
- [SMPL] [ICCV'23] [ReMoDiffuse](https://github.com/MotrixLab/ReMoDiffuse): Retrieval-Augmented Motion Diffusion Model.
- [SMPL] [TPAMI'24] [MotionDiffuse](https://github.com/MotrixLab/MotionDiffuse): Text-Driven Human Motion Generation with Diffusion Model.

### Motion Dataset
- [SMPL] [ECCV'22] [HuMMan](https://github.com/MotrixLab/humman_toolbox): Toolbox for HuMMan, a large-scale multi-modal 4D human dataset.
- [SMPLX] [T-PAMI'24] [GTA-Human](https://github.com/MotrixLab/gta-human_toolbox): Toolbox for GTA-Human, a large-scale 3D human dataset generated with the GTA-V game engine.
