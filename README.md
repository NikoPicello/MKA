# Markerless Kinematic Analysis (MKA)

We introduce Markerless Kinematic Analysis (MKA), an end-to-end framework that reconstructs full-body, articulated 3D meshes, including hands and manipulated objects, from ordinary RGB videos captured with single or multiple consumer cameras. 
Multi-view fusion and explicit human-object interaction modeling yield anatomically consistent, metric-scale poses that generalize to cluttered homes, gyms, and clinics. 
By merging computer vision with rehabilitative medicine, MKA enables continuous, objective, and scalable motion monitoring in natural environments, opening avenues for personalized training, tele-rehabilitation, and population-level musculoskeletal health surveillance.


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


##  Running the Pipeline

To execute the full processing pipeline:
```bash
bash run_pipeline.sh
```
