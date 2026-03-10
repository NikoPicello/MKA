#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh

PROJECT_NAME=004096

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYOPENGL_PLATFORM=osmesa

IN_ROOT=sample_video/${PROJECT_NAME}/lego
OUT_ROOT=results/${PROJECT_NAME}/lego

# conda activate sam2
# echo "=== running sam2"
# python scripts/sam_tracking.py \
#     --video_dir ${IN_ROOT} \
#     --prompt_file sample_video/sam_prompt.json \
#     --out_dir ${OUT_ROOT}/object
# 
# conda deactivate
conda activate mka


smplestx_out_dir=${OUT_ROOT}/smplestx
echo "=== running smplestx"
python scripts/smplestx_multiview.py \
    --video_dir ${IN_ROOT} --out_dir ${smplestx_out_dir}

kpt3d_out_dir=${OUT_ROOT}/kpt3d
echo "=== running triangulation"
python scripts/triangulate_multiview.py \
    --video_dir ${IN_ROOT} \
    --kpt2d_dir ${smplestx_out_dir} \
    --out_dir ${kpt3d_out_dir} 

apose_out_dir=${OUT_ROOT}/smplify/apose
echo "=== running smplify on first 10 frames to estimate shape"
python scripts/smplify_xrmocap_mka.py \
    --config configs/mview_sperson_smplify_apose.py \
    --kps3d_file ${kpt3d_out_dir}/optim_kpt3d.npz \
    --cam_file ${IN_ROOT}/cam_info.json \
    --output_dir ${apose_out_dir} \
    --video_dir ${IN_ROOT}  \
    --src_convention smplx --tgt_convention smplx \
    --model smplx --vis_smpl \
    --start_t 0 --end_t 10 --frame_interval 1 

action_out_dir=${OUT_ROOT}/smplify/action
echo "=== running smplify on whole sequence"
python scripts/smplify_xrmocap_mka.py \
    --config configs/mview_sperson_smplify_action.py \
    --init_smpl_file ${apose_out_dir}/human_data_tri_smplx.npz \
    --kps3d_file ${kpt3d_out_dir}/optim_kpt3d.npz \
    --cam_file ${IN_ROOT}/cam_info.json \
    --output_dir ${action_out_dir} \
    --video_dir ${IN_ROOT}  \
    --src_convention smplx --tgt_convention smplx \
    --model smplx --vis_smpl --frame_interval 1 

hamer_out_dir=${OUT_ROOT}/hamer
echo "=== run hamer"
python scripts/hamer_infer_multiview.py \
    --video_dir ${IN_ROOT} \
    --out_dir ${hamer_out_dir} 

echo "=== pack with mano"
python scripts/pack_mano_smplx.py \
    --video_dir ${IN_ROOT} \
    --data_dir ${OUT_ROOT} \
    --out_dir ${OUT_ROOT}/pack \
    --use_mano 

echo "=== pack with mano and sam"
python scripts/pack_mano_smplx_sam.py \
    --video_dir ${IN_ROOT} \
    --data_dir ${OUT_ROOT} \
    --sam_dir ${OUT_ROOT}/object \
    --out_dir ${OUT_ROOT}/pack \
    --use_mano 

conda deactivate 
