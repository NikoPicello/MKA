#output_root="/home/caizhongang/github/zoehuman/data_temp/renbody/sensebee_datalist_83155/20220424/zhengwei_m/zhengwei_yf2_dz9"
#annots_path="/home/caizhongang/github/zoehuman/data_temp/renbody/sensebee_datalist_83155/20220424/zhengwei_m/zhengwei_yf2_dz9/annots.npy"
#background_dir="/home/caizhongang/github/zoehuman/data_temp/renbody/sensebee_datalist_83155/20220424/zhengwei_m/zhengwei_yf2_dz9/image/25"
#gender="male"
#vis_view=25

output_root="/home/caizhongang/github/zoehuman/data_temp/renbody/sensebee_datalist_80635/yahongyue_f_apose_yf1"
annots_path="/home/caizhongang/github/zoehuman/data_temp/renbody/sensebee_datalist_80636/annots.npy"
background_dir="/home/caizhongang/github/zoehuman/data_temp/renbody/sensebee_datalist_80636/yahongyue_f_apose_yf1/image/25"
gender="female"
vis_view=25

echo "smplifyx"
python tools/ren_body/smplifyx.py \
        --vis_smpl \
        --config configs/smplify/ren_body.py \
        --kp3d_path ${output_root}/pose_3d/optim/human_data_tri.npz \
        --output_folder ${output_root}/smplx_${gender} \
        --model "smplx" \
        --gender ${gender} \
        --src_convention "human_data" \
        --tgt_convention "openpose_118" \
        --save_mesh

echo "visualize"
python tools/visualization/visualize_project_smplx.py \
        --smplx_path ${output_root}/smplx_${gender}/human_data_tri_smplx.npz \
        --annots_path ${annots_path} \
        --view ${vis_view}\
        --output_video_path ${output_root}/smplx_${gender}/human_data_tri_smplx_view${vis_view}.mp4 \
        --background_dir ${background_dir}
