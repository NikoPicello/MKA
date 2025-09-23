import os
import numpy as np
import mmcv
import torch
from xrmocap.model.body_model.builder import build_body_model
from pytorch3d.structures import Meshes
from pytorch3d.io import IO
from pytorch3d.renderer.mesh.textures import TexturesVertex
from tqdm import tqdm
from mmhuman3d.utils.ffmpeg_utils import images_to_video

def main():
    apose = False
    is_test = False
    # train_list = '/mnt/cache/yinwanqi/01-project/zoehuman/data/data_list/ablation.txt'
    train_list = '/mnt/lustre/share_data/chengwei/RenBody/benchmark_splits/train/matrix2.txt'
    test_list = '/mnt/cache/yinwanqi/01-project/zoehuman/data/data_list/new_test_230207.txt'
    seq_output_root = '/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full'
    apose_output_root = '/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full_apose'
    image_root = '/mnt/cache/yinwanqi/01-project/zoehuman/data/RenBody'

    if is_test:
        input_list = test_list
        state = 'test'
    else: 
        input_list = train_list
        state = 'train'
    file = open(input_list,'r')
    lines_raw = file.readlines()
    train_person_list = [line.strip('\n') for line in lines_raw]
    for i, person_folder in enumerate(train_person_list):
        # if i != 6:
        #     continue
        print(f'{state}-{i}: {person_folder}')
        person_folder_one = person_folder.replace('/', '_')
        # import pdb;pdb.set_trace()
        
        # update gender and dir
        g = person_folder.split('/')[1][-1]
        if g == 'f':
            gender = 'female'
        elif g =='m':
            gender = 'male'
        
        if not apose:
            smplx_dir = os.path.join(seq_output_root,person_folder,'smplx_xrmocap','human_data_tri_smplx.npz')
            save_smplx_dir = os.path.join(seq_output_root,person_folder,'smplx_xrmocap','human_data_smplx.npz')
            output_dir_mesh = os.path.join(seq_output_root,person_folder, 'mesh')
            output_dir_video = os.path.join(seq_output_root,'raw_video')
            mesh_only_dir = os.path.join(seq_output_root, 'mesh_matrix',person_folder_one)
            train_mesh_dir = os.path.join(seq_output_root, 'mesh_train',person_folder_one)
            
            if os.path.exists(train_mesh_dir):
                print(f'!!!!{train_mesh_dir} exists, skip')
                continue
                
            if os.path.exists(mesh_only_dir):
                print(f'~~~~~{mesh_only_dir} exists, skip')
                continue
                
            os.makedirs(mesh_only_dir, exist_ok=False)
            # os.makedirs(output_dir_mesh, exist_ok=False)
            # os.makedirs(output_dir_video, exist_ok=False)

            # image_input = os.path.join(image_root,person_folder,'image','25')
            # video_save_name = '_'.join(person_folder.split('/')) 
            # video_save_path = os.path.join(output_dir_video,video_save_name+'.mp4')

            # load smplx npz
            # smpl_dict_raw = dict(np.load(smplx_dir))

            # smpl_dict = smpl_dict_raw.copy()
            # fullpose = smpl_dict.pop('fullpose')
            # n_frame = fullpose.shape[0]
            # # import pdb;pdb.set_trace()
            # smpl_dict['global_orient'] = fullpose[:, 0].reshape(n_frame, 3)
            # smpl_dict['body_pose'] = fullpose[:, 1:22].reshape(n_frame, 63)
            # smpl_dict['jaw_pose'] = fullpose[:, 22].reshape(n_frame, 3)
            # smpl_dict['leye_pose'] = fullpose[:, 23].reshape(n_frame, 3)
            # smpl_dict['reye_pose'] = fullpose[:, 24].reshape(n_frame, 3)
            # smpl_dict['left_hand_pose'] = fullpose[:, 25:40].reshape(n_frame, 45)
            # smpl_dict['right_hand_pose'] = fullpose[:, 40:55].reshape(n_frame, 45)
            # smpl_dict['body_pose'] = fullpose[:, 1:22].reshape(n_frame, 63)
            # smpl_dict['expression'] = np.zeros((n_frame,10))

            # np.savez(save_smplx_dir, **smpl_dict)
            # print(f"saved to {save_smplx_dir}")

            smpl_dict = dict(np.load(save_smplx_dir))
        else:
            date, person_id = person_folder.split('/')[:2]
            yf = person_folder.split('/')[-1].split('_')[-2]
            name = person_folder.split('/')[1][:-2]
            apose_person_folder = f'{name}_apose_{yf}'
            # print(f'apose folder: {apose_person_folder}')

            smplx_dir = os.path.join(apose_output_root,date,person_id,apose_person_folder,'smplx_xrmocap','human_data_tri_smplx.npz')
            output_dir_mesh = os.path.join(apose_output_root,date,person_id,apose_person_folder, 'mesh')
            os.makedirs(output_dir_mesh, exist_ok=True)
            print(f'smpl folder: {smplx_dir}')

            # load smplx npz
            smpl_dict = {}
            smpl_dict_raw = dict(np.load(smplx_dir))

            smpl_dict = smpl_dict_raw.copy()
            fullpose = smpl_dict.pop('fullpose')
            n_frame = fullpose.shape[0]
            # import pdb;pdb.set_trace()
            smpl_dict['global_orient'] = fullpose[:, 0].reshape(n_frame, 3)
            smpl_dict['body_pose'] = fullpose[:, 1:22].reshape(n_frame, 63)
            smpl_dict['jaw_pose'] = fullpose[:, 22].reshape(n_frame, 3)
            smpl_dict['leye_pose'] = fullpose[:, 23].reshape(n_frame, 3)
            smpl_dict['reye_pose'] = fullpose[:, 24].reshape(n_frame, 3)
            smpl_dict['left_hand_pose'] = fullpose[:, 25:40].reshape(n_frame, 45)
            smpl_dict['right_hand_pose'] = fullpose[:, 40:55].reshape(n_frame, 45)
            smpl_dict['body_pose'] = fullpose[:, 1:22].reshape(n_frame, 63)
            smpl_dict['expression'] = np.zeros((n_frame,10))
        
        # from image folder to video
        # images_to_video(input_folder = image_input,
        #             output_path = video_save_path,
        #             remove_raw_file = False,
        #             img_format = '%05d.jpg',
        #             fps = 25)
        
        # import pdb; pdb.set_trace()

        # # init body model
        body_model_config = mmcv.Config.fromfile('/mnt/cache/yinwanqi/01-project/zoehuman/configs/smplify/save_mesh.py')
        body_model_config.body_model.update(dict(gender=gender))
        body_model = build_body_model(dict(body_model_config.body_model)).to('cuda')

        # # pass params to body model
        body_model_kwargs = dict(return_verts=True)
        for key, val in smpl_dict.items():
            if key in ['betas', 'global_orient', 'body_pose', 'left_hand_pose',
                'right_hand_pose', 'transl', 'expression', 'jaw_pose',
                'leye_pose', 'reye_pose']:
                val = torch.Tensor(val).to('cuda')
                smpl_dict[key] = val
        body_model_kwargs.update(smpl_dict)
        # import pdb; pdb.set_trace()
        body_model_output = body_model(**body_model_kwargs)

        # # save mesh
        face = body_model.faces
        if isinstance(face, np.ndarray):
            face = torch.from_numpy(face.astype(np.int32))
        vertices = torch.tensor(body_model_output['vertices']).clone().cpu()

        for frame_index in tqdm(range(vertices.shape[0])):
        #  vertices.shape: [1, 10475, 3]
        #  faces.shape: [20908, 3]
            meshes = Meshes(
                verts=vertices[frame_index:frame_index + 1, :, :],
                faces=face.view(1, -1, 3),
                textures=TexturesVertex(
                    verts_features=torch.FloatTensor((
                        1, 1, 1)).view(1, 1, 3).repeat(
                        1, vertices.shape[-2], 1)))
            mesh_path = os.path.join(mesh_only_dir,
                                        f'{frame_index:06d}.obj')
            IO().save_mesh(data=meshes, path=mesh_path)

        print(f"Mesh saved as {mesh_only_dir}")

    return True

if __name__ == '__main__':

    success = main()
    # if not success:
        # print(f">>>Not success ({e}): {seq}")