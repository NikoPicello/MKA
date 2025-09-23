import argparse
import os
import shutil
import json
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from mmhuman3d.utils.ffmpeg_utils import images_to_video
import subprocess
from mmhuman3d.core.visualization.visualize_smpl import \
    visualize_smpl_calibration
from xrmocap.core.visualization.visualize_keypoints3d import visualize_project_keypoints3d
from xrmocap.data_structure.keypoints import Keypoints
from xrprimer.data_structure.camera import FisheyeCameraParameter
import mmcv
from xrmocap.model.body_model.builder import build_body_model

def parse_args():
    parser = argparse.ArgumentParser('SMPLify tools')
    # path
    parser.add_argument('--subset', type=str,
        default='/mnt/cache/yinwanqi/01-project/zoehuman/data/data_list/extra.txt')
    parser.add_argument('--img', type=str,
        default='/mnt/cache/yinwanqi/01-project/zoehuman/data/RenBody')
    parser.add_argument('--smplx', type=str,
        default='/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full')
    parser.add_argument('--kps3d', type=str,
        default='/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full')
    parser.add_argument('--msk', type=str,
        default='/mnt/cache/yinwanqi/01-project/zoehuman/data/RenBody_mask_test')
    parser.add_argument('--dst', type=str,
        default='/mnt/cache/yinwanqi/01-project/zoehuman/data/RenBody_masked')
    parser.add_argument('--cam', type=str,
        default='25')
    args = parser.parse_args()
    return args

def get_camera_param_from_annots(annots_file, view):
    ren_body_cam_dict = np.load(
    annots_file, allow_pickle=True).item()['cams']
    
    camera_para_dict = {
        'RT': ren_body_cam_dict[view]['RT'].reshape(1, 4, 4),
        'K': ren_body_cam_dict[view]['K'].reshape(1, 3, 3),
        } if '00' in ren_body_cam_dict.keys() else \
        {
        'RT': ren_body_cam_dict['RT'][int(view)].reshape(1, 4, 4),
        'K': ren_body_cam_dict['K'][int(view)].reshape(1, 3, 3),
        }

    extrinsic = camera_para_dict['RT'][0, :, :]  # 4x4 mat
    intrinsic = camera_para_dict['K'][0, :, :]  # 3x3 mat
    r_mat_inv = extrinsic[:3, :3]
    r_mat = np.linalg.inv(r_mat_inv)
    t_vec = extrinsic[:3, 3:]
    t_vec = -np.dot(r_mat, t_vec).reshape((3))

    dist_array = ren_body_cam_dict[view]['D'] \
            if '00' in ren_body_cam_dict.keys() else \
            np.array(ren_body_cam_dict['D'][int(view)])
    dist_keys = ['k1', 'k2', 'p1', 'p2', 'k3']
    dist_coeff_k =[]
    dist_coeff_p = []

    distortion_coefficients = {}
    for dist_index, dist_key in enumerate(dist_keys):
        if 'k' in dist_key:
            dist_coeff_k.append(dist_array[dist_index])
        if 'p' in dist_key:
            dist_coeff_p.append(dist_array[dist_index])
        distortion_coefficients[dist_key] = float(dist_array[dist_index])

    return intrinsic, r_mat, t_vec, dist_coeff_k, dist_coeff_p, distortion_coefficients

if __name__ == '__main__':
    args = parse_args()
    
    file = open(args.subset,'r')
    lines_raw = file.readlines()
    lines_sub = [line.strip('\n') for line in lines_raw]

    log = lines_sub.copy()
    for i, l in enumerate(lines_sub):
        print(f"{i}/{len(lines_sub), l}***************")
        video_name = l.replace("/","_")

        # ##            
        # image_dir = os.path.join(args.img, l, 'image', str(args.cam))
        # # mask_dir = os.path.join(args.msk, l, 'mask', str(args.cam))
        # save_dir = os.path.join(args.dst, 'image', l, str(args.cam))
        # video_dir = os.path.join(args.dst, 'video', f'{video_name}.mp4')
        # raw_video_dir = os.path.join(args.dst, 'raw_video', f'{video_name}.mp4')

        # os.makedirs(save_dir, exist_ok=True)

        # image_list = os.listdir(image_dir)
        # # mask_list = os.listdir(mask_dir)

        # image_list.sort()
        # mask_list.sort()
        # import pdb; pdb.set_trace()
        # images_to_video(
        #     input_folder=save_dir,
        #     output_path=video_dir,
        #     remove_raw_file=False,
        #     img_format='%05d.jpg',
        #     fps=30)
        
        # images_to_video(
        #     input_folder=image_dir,
        #     output_path=raw_video_dir,
        #     remove_raw_file=False,
        #     img_format='%05d.jpg',
        #     fps=30)
        
        # save original video
        # raw_img_dir = os.path.join(args.img, l, 'image', args.cam, '%05d.jpg')
        # video_dir = os.path.join(args.dst, 'raw_video', video_name+'.mp4')
        # mov_command = [
        #     'ffmpeg',
        #     '-y',
        #     '-i',
        #     f'{raw_img_dir}', 
        #     '-vcodec',
        #     'libx264',
        #     f'{video_dir}',
        # ]
        # print(f'Running \"{" ".join(mov_command)}\"')
        # subprocess.check_call(mov_command)

        # import pdb; pdb.set_trace()

        # process mask
        # image_dir = os.path.join(args.img, l, 'image', str(args.cam))
        # mask_dir = os.path.join(args.msk, l, 'mask', str(args.cam))
        # save_dir = os.path.join(args.dst, 'image', l, str(args.cam))
        # video_dir = os.path.join(args.dst, 'video', video_name)

        # os.makedirs(save_dir, exist_ok=True)

        # image_list = os.listdir(image_dir)
        # mask_list = os.listdir(mask_dir)

        # image_list.sort()
        # mask_list.sort()
        
        # for j, image_name in enumerate(tqdm(image_list)):
        #     # import pdb; pdb.set_trace()
        #     image_to_load = os.path.join(image_dir, image_name)
        #     mask_to_load = os.path.join(mask_dir, mask_list[j])
        #     image_name = image_name[:-3]  + 'png'
        #     image_to_save = os.path.join(save_dir, image_name)

        #     image = cv2.imread(image_to_load)
        #     mask = cv2.imread(mask_to_load)
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        #     x, y = image.shape[:2]
        #     x_, y_ = mask.shape[:2]

        #     if not (x == x_ and y == y_):
        #         mask = Image.fromarray(mask)
        #         mask = mask.resize((x, y))
        #         mask = np.array(mask) 
        #     if mask.shape[2] == 3:
        #         mask = mask[..., 0]

        #     image[..., -1] = mask
        #     cv2.imwrite(image_to_save, image)

        # prepare camera
        annots_file_dir = os.path.join(args.img, l.split('/')[0], l.split('/')[1],'annots.npy')
        height = 2448
        width = 2048
        camera_parameter = FisheyeCameraParameter(name=args.cam)
        K, R, T, dist_coeff_k, dist_coeff_p, dist_coeff_dict = \
            get_camera_param_from_annots(annots_file_dir, args.cam)
        camera_parameter.set_KRT(K, R, T)
        camera_parameter.set_dist_coeff(dist_coeff_k,dist_coeff_p)
        camera_parameter.inverse_extrinsic()
        camera_parameter.set_resolution(height, width) # height, width

        K = np.asarray(camera_parameter.get_intrinsic())
        R = np.asarray(camera_parameter.get_extrinsic_r())
        T = np.asarray(camera_parameter.get_extrinsic_t())

        # load smplx
        smplx_dir = os.path.join(args.smplx, l, 'smplx_xrmocap', 'human_data_tri_smplx.npz')
        smplx_data = dict(np.load(smplx_dir))
        smplx_video_dir = os.path.join(args.dst, 'smplx', video_name+'.mp4')

        g = l.split('/')[1][-1]
        if g == 'f':
            gender = 'female'
        elif g =='m':
            gender = 'male'

        n_frame = smplx_data['betas'].shape[0]
        # import pdb; pdb.set_trace()

        fullpose = smplx_data['fullpose'].reshape(n_frame, -1)
        transl = smplx_data['transl']
        betas = smplx_data['betas']

        # init body model
        body_model_config_update = dict(model_path='mmhuman3d/data/body_models')
        body_model_config = mmcv.Config.fromfile('/mnt/cache/yinwanqi/01-project/zoehuman/configs/smplify/save_mesh.py')
        
        body_model_config.body_model.update(dict(gender=gender))
        body_model_config.body_model.update(body_model_config_update)
        # body_model = build_body_model(dict(body_model_config.body_model)).to('cuda')
        
        # load keypoints3d
        keypoints3d_dir = os.path.join(args.kps3d, l, 'smplx_xrmocap', 'human_data_optimized_xr_keypoints3d.npz')
        keypoints3d_data = Keypoints().fromfile(keypoints3d_dir)
        keypoints3d_video_data = os.path.join(args.dst, 'kps3d', video_name+'.mp4')
        
        # save keypoints3d video
        white_array = np.ones((n_frame, height, width, 3)) * 255
        tri_res = visualize_project_keypoints3d(
            keypoints=keypoints3d_data,
            cam_param=camera_parameter,
            output_path=keypoints3d_video_data,
            img_arr=white_array,
            overwrite = True,
            return_array=False)
   
        # save smplx reprojection video
        white_array = np.ones((n_frame, height, width, 3)) * 255
        results = visualize_smpl_calibration(
            poses=fullpose,
            betas=betas,
            transl=transl,
            K=K,
            R=R,
            T=T,
            overwrite=True,
            body_model_config=body_model_config.body_model,
            output_path=smplx_video_dir,
            image_array=white_array,
            resolution=(height, width),
            return_tensor=True,
            alpha=1.0,
            batch_size=5,
            plot_kps=False,
            vis_kp_index=False)



        # # if i != 14:
        # #     continue
        # print(f"{i}/{len(lines_sub), l}***************")
        # # src_root = "/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full"
        # # dst_root = "/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full/mesh_test"
        # video_name = l.replace("/","_")
        # # src = os.path.join(args.src, l, 'mesh')
        # # dst = os.path.join(args.dst, folder_name)
        # # video_name = l.replace("/","_")
        # # src = os.path.join(args.src, video_name+".mp4")
        # # dst = os.path.join(args.dst, video_name+".mp4")
        # # import pdb; pdb.set_trace()
        # ### apose 
        # for args.cam in range(60):
        #     print(args.cam)
        #     args.cam = f'{args.cam:02d}'
        #     image_dir = os.path.join(args.img, l, 'image', str(args.cam))
        #     mask_dir = os.path.join(args.msk, l, 'mask', str(args.cam))
        #     save_dir = os.path.join(args.dst, 'image', l, str(args.cam))
        #     video_dir = os.path.join(args.dst, 'video', video_name)

        #     os.makedirs(save_dir, exist_ok=True)

        #     image_list = os.listdir(image_dir)
        #     mask_list = os.listdir(mask_dir)

        #     image_list.sort()
        #     mask_list.sort()
            
        #     for j, image_name in enumerate(tqdm(image_list)):
        #         # import pdb; pdb.set_trace()
        #         image_to_load = os.path.join(image_dir, image_name)
        #         mask_to_load = os.path.join(mask_dir, mask_list[j])
        #         image_name = image_name[:-3]  + 'png'
        #         image_to_save = os.path.join(save_dir, image_name)

        #         image = cv2.imread(image_to_load)
        #         mask = cv2.imread(mask_to_load)
        #         image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        #         x, y = image.shape[:2]
        #         x_, y_ = mask.shape[:2]

        #         if not (x == x_ and y == y_):
        #             mask = Image.fromarray(mask)
        #             mask = mask.resize((x, y))
        #             mask = np.array(mask) 
        #         if mask.shape[2] == 3:
        #             mask = mask[..., 0]

        #         image[..., -1] = mask
        #         cv2.imwrite(image_to_save, image)

        # save png to gif, mov
        # gif_command_1 = [
        #     'ffmpeg', 
        #     '-y',
        #     '-i', 
        #     f'{save_dir}/%05d.png', 
        #     '-vf',
        #     'palettegen',
        #     f'{video_dir}_palette.png',
        #     # '&&',
        #     # 'ffmpeg',
        #     # '-i',
        #     # f'{save_dir}/%05d.png',
        #     # '-i',
        #     # f'{video_dir}_palette.png',
        #     # '-lavfi', 
        #     # '"paletteuse"',
        #     # '-y',
        #     # f'{video_dir}.gif',
        # ]
        # gif_command_2 = [
        #     'ffmpeg',
        #     '-y',
        #     '-i',
        #     f'{save_dir}/%05d.png',
        #     '-i',
        #     f'{video_dir}_palette.png',
        #     '-lavfi', 
        #     'paletteuse',
        #     '-y',
        #     f'{video_dir}.gif',
        # ]

        # mov_command = [
        #     'ffmpeg',
        #     '-y',
        #     '-i',
        #     f'{save_dir}/%05d.png', 
        #     '-vcodec',
        #     'png',
        #     f'{video_dir}.mov',
        # ]
        # # import pdb; pdb.set_trace()
        # print(f'Running \"{" ".join(gif_command_1)}\"')
        # subprocess.check_call(gif_command_1)
        # print(f'Running \"{" ".join(gif_command_2)}\"')
        # subprocess.check_call(gif_command_2)
        # os.remove(f'{video_dir}_palette.png')

        # # import pdb; pdb.set_trace()
        # print(f'Running \"{" ".join(mov_command)}\"')
        # subprocess.check_call(mov_command)

        


    #     folder_name = l.replace("/","_")
    #     date, person_id = l.split('/')[:2]
    #     yf = l.split('/')[-1].split('_')[-2]
    #     name = l.split('/')[1][:-2]
    #     apose_person_folder = f'{name}_apose_{yf}'
    #     # print(f'apose folder: {apose_person_folder}')
    #     apose_output_root = '/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full_apose'
    #     src = os.path.join(apose_output_root,date,person_id,apose_person_folder, 'mesh')
    #     dst = os.path.join(args.dst, folder_name)
    #     print(f'Copying \n{src} to \n{dst}')
    #     date_m = int(l.split('_')[0][4:6])
    #     # if date_m < 7:
    #     #     continue
    #     if not os.path.exists(src):
    #         print("no mesh folder", l)
    #         continue
    #     if args.copy:
    #         if not os.path.exists(args.dst):
    #                 os.makedirs(args.dst)
    #         # shutil.copyfile(src, dst)
    #         shutil.copytree(src, dst)
    #         # if os.path.isfile(src):
    #         #     os.remove(src)
    #         # else:
    #         #     print(src)
    #     log.remove(l)

    # print(log)

    # print("#seq in subset: ",len(lines))
    # print("#seq in successful list: ",len(lines_s))
    # print("#seq in subset but not processed successfully: ", len(lines_c))
    # print("#seq remain in processed list: ", len(lines_s_c))

    # file.close()
    # file_success.close()
    
    # print("lack of these sequences, check reasons:")
    # for l in lines_c:
    #     print(l)
    
    # # print("extras:")
    # # for l in lines_s_c:
    # #     print(l)
    
    # jason
    # failure_list={}
    # with open(args.subset) as json_file:
    #     data = json.load(json_file)
    #     data.pop("definition")
    #     for category, data_list in data.items():
    #         log = data_list.copy()
    #         for sequence in data_list:
    #             video_name = sequence.replace("/","_")
    #             src = os.path.join(args.src, video_name+".mp4")
                
    #             dst_dir = os.path.join(args.dst, category)
    #             if not os.path.exists(dst_dir):
    #                 os.makedirs(dst_dir)
    #             dst = os.path.join(dst_dir, video_name+".mp4")
                
    #             print(f'Copying {src} to {dst}')
    #             if args.copy:
    #                 shutil.copyfile(src, dst)
    #             log.remove(sequence)
            
    #         failure_list[category] = log
        
    
    # print(failure_list)

    # failure_list={}
    # with open(args.subset) as json_file:
    #     data = json.load(json_file)
    #     data.pop("definition")
    #     for category, data_list in data.items():
    #         log = data_list.copy()
    #         for sequence in data_list:
    #             video_name = sequence.replace("/","_")
    #             src = os.path.join(args.src, video_name+".mp4")
                
    #             dst_dir = os.path.join(args.dst, category)
    #             if not os.path.exists(dst_dir):
    #                 os.makedirs(dst_dir)
    #             dst = os.path.join(dst_dir, video_name+".mp4")
                
    #             print(f'Copying {src} to {dst}')
    #             if args.copy:
    #                 shutil.copyfile(src, dst)
    #             log.remove(sequence)
            
    #         failure_list[category] = log
        
    
    # print(failure_list)

    
