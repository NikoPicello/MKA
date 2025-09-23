import numpy as np
import os
from beautifultable import BeautifulTable
from xrprimer.ops.projection.opencv_projector import OpencvProjector
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrmocap.transform.convention.keypoints_convention import convert_keypoints
from xrmocap.data_structure.keypoints import Keypoints
from mmhuman3d.data.data_structures.human_data import HumanData
from xrmocap.core.visualization.visualize_keypoints2d import visualize_keypoints2d
from xrmocap.transform.convention.keypoints_convention import ( 
    get_keypoint_idx, get_keypoint_idxs_by_part,
)
from tqdm import tqdm
import csv
import cv2
from datetime import datetime

root3d = '/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full'
root2d = '/mnt/cache/yinwanqi/01-project/zoehuman/data/RenBody'
# subset_root = '/mnt/cache/yinwanqi/01-project/zoehuman/data/benchmark_splits/splits'
subset_root = '/mnt/cache/yinwanqi/01-project/zoehuman/data/data_list'
subset_root_tt = '/mnt/cache/yinwanqi/01-project/zoehuman/data/data_list'
log_root = '/mnt/cache/yinwanqi/01-project/zoehuman/data/data_list'

# set_list = ['motion_simple',
#             'motion_medium',
#             'motion_hard',
#             'texture_simple',
#             'texture_medium',
#             'texture_hard',
#             'deformation_simple',
#             'deformation_medium',
#             'deformation_hard',
#             'interaction_simple',
#             'interaction_medium',
#             'interaction_hard',
#             'interaction_no']

# set_list = ['texture_medium',
#             'interaction_hard',]
# set_list = ['new_train_final', 'new_test_final']        
set_list_tt = ['new_train_230207',]            

def get_part_mask(keypoints, part_name):
    convention = keypoints.get_convention()
    mask = keypoints.get_mask()

    if part_name == 'left_hand':
        left_hand_keypoint_idxs = get_keypoint_idxs_by_part(
            'left_hand', convention=convention)
        target_keypoint_idxs = left_hand_keypoint_idxs
    elif part_name == 'right_hand':
        right_hand_keypoint_idxs = get_keypoint_idxs_by_part(
            'right_hand', convention=convention)
        target_keypoint_idxs = right_hand_keypoint_idxs
    elif part_name == 'hand':
        left_hand_keypoint_idxs = get_keypoint_idxs_by_part(
            'left_hand', convention=convention)
        right_hand_keypoint_idxs = get_keypoint_idxs_by_part(
            'right_hand', convention=convention)
        target_keypoint_idxs = [
            *left_hand_keypoint_idxs, *right_hand_keypoint_idxs
        ]
    elif part_name == 'body':
        # import pdb; pdb.set_trace()
        target_keypoint_idxs = get_keypoint_idxs_by_part(
            'body', convention=convention)
    
    part_mask = np.zeros((1, mask.shape[2]))
    part_mask[0, target_keypoint_idxs] = 1.0
    processed_mask = np.multiply(mask, part_mask)  
    return processed_mask

def calc_mpjpe(joints_keypoints, kps_keypoints, kps_dim=3, part_name=None, offset=False):
    joints = joints_keypoints.get_keypoints()[..., :kps_dim]
    keypoints = kps_keypoints.get_keypoints()[..., :kps_dim]
    convention = kps_keypoints.get_convention()
    
    if part_name is None:
        mask = kps_keypoints.get_mask()
    else: 
        mask = get_part_mask(kps_keypoints, part_name)
    
    n_frame = mask.shape[0]
    mpjpe = []
    
    if offset:
        left_writst_id = get_keypoint_idx('left_wrist', convention = convention)
        right_writst_id = get_keypoint_idx('right_wrist', convention = convention)

        for i, frame in enumerate(range(n_frame)):
            # import pdb; pdb.set_trace()
            left_offset, right_offset = joints[frame, 0, [left_writst_id, right_writst_id], :] - \
                                        keypoints[frame, 0, [left_writst_id, right_writst_id], :]
            left_mask = get_part_mask(kps_keypoints, part_name='left_hand')
            right_mask = get_part_mask(kps_keypoints, part_name='right_hand')
            joints[frame, 0, np.where(left_mask[frame, 0, :]>0), :] -= left_offset
            joints[frame, 0, np.where(right_mask[frame, 0, :]>0), :] -= right_offset
            

    for i, frame in enumerate(range(n_frame)):
        
        pred_kps3d = joints[frame, 0, np.where(mask[frame, 0, :]>0), :]
        gt_kps3d = keypoints[frame, 0, np.where(mask[frame, 0, :]>0), :]

        # import pdb; pdb.set_trace()
        if np.isnan(gt_kps3d).any():
            print("warning: check mask, some of the nan kpts are not masked properly.")
        mpjpe_value = np.sqrt(np.sum(np.square(pred_kps3d - gt_kps3d), axis=-1)).mean()
        
        if not np.isnan(mpjpe_value):
            # exclude the frame if the view is not used for this frame
            mpjpe.append(mpjpe_value)

    mpjpe_mean = np.array(mpjpe).mean()
    return mpjpe_mean


def calc_repj(camera_parameter_list, human_data_list, keypoints, background_dir, seq_name, sort=False):
    # human_data convention
    rej_all_cam = []
    for i, cam in enumerate(camera_parameter_list):
        # reproject keypoints3d
        projected_kps2d_full = np.ones((keypoints.get_frame_number(),
                                              keypoints.get_person_number(),
                                              keypoints.get_keypoints_number(),
                                              3))
        projector = OpencvProjector(camera_parameters=[cam])
        projected_kps2d = projector.project(
        points=keypoints.get_keypoints()[..., :3].reshape(-1, 3),
        points_mask=np.expand_dims(keypoints.get_mask(), axis=-1))
        projected_kps2d = projected_kps2d.reshape(keypoints.get_frame_number(),
                                              keypoints.get_person_number(),
                                              keypoints.get_keypoints_number(),
                                              2)
        projected_kps2d_full[..., :2] = projected_kps2d

        # get kps2d
        keypoints2d = human_data_list[i]
        projected_keypoints2d =  Keypoints(            
            dtype='numpy',
            kps=projected_kps2d_full,
            mask=keypoints2d.get_mask(),
            convention='human_data')

        # calculate error 2d
        assert projected_keypoints2d.get_convention() == 'human_data'
        assert keypoints2d.get_convention() == 'human_data'
        repj_value = calc_mpjpe(projected_keypoints2d, keypoints2d, kps_dim=2)
        if np.isnan(repj_value):
            print(f'cam {i} not used at all in seq {seq_name}')
        else:
            rej_all_cam.append(repj_value)
        

        # visualize
        # output_path = '/mnt/cache/yinwanqi/01-project/zoehuman/data/data_list'
        # if i in [10,25,45]:
        #     image_array, _ = get_image_array(background_dir, str(i))
        #     import pdb; pdb.set_trace()
        #     proj_array = visualize_keypoints2d(
        #         keypoints=projected_keypoints2d,
        #         output_path=os.path.join(output_path,f'keypoints2d_proj_view{i}.mp4'),
        #         img_arr=image_array,
        #         overwrite=True,
        #         return_array=True)
            
        #     _ = visualize_keypoints2d(
        #         keypoints=keypoints2d,
        #         output_path=os.path.join(output_path,f'keypoints2d_view{i}.mp4'),
        #         img_arr=proj_array,
        #         overwrite=True,
        #         return_array=True)
    # import pdb; pdb.set_trace()
    rej_all_cam_array = np.array(rej_all_cam)
    if (rej_all_cam_array>500).any():
        print('ooops')
    rej_all_cam_mean = rej_all_cam_array.mean()
    # rej_all_cam_mean = rej_all_cam_array[~np.isnan(rej_all_cam_array)].mean()
    if sort:
        rej_all_cam.sort()
    
    return rej_all_cam_mean, rej_all_cam

def calc_stats(human_data_list):
    view_number = len(human_data_list)
    # import pdb; pdb.set_trace()
    frame_number, _, keypoints_number, _ = \
            human_data_list[0].get_keypoints().shape
    keypoints2d_np = np.zeros(
            shape=(view_number, frame_number, keypoints_number, 3))

    for i, keypoints2d in enumerate(human_data_list):
        keypoints2d_np[i, ...] = keypoints2d.get_keypoints().squeeze()

    keypoints2d_mask = human_data_list[0].get_mask()[0].squeeze()

    kps_thr, valid_cam_array = __try_keypoints_threshold__(keypoints2d_np, keypoints2d_mask)

    valid_num_cam = [valid_cam_array.min(), valid_cam_array.max(), int(valid_cam_array.mean())]

    ignore_idxs = np.where(
                keypoints2d_np[:, :, :, 2] < kps_thr)
    keypoints2d_np[ignore_idxs[0], ignore_idxs[1],
                       ignore_idxs[2], :] = np.nan
    mask_idxs = np.where(keypoints2d_mask[:] == 0)
    keypoints2d_np[:, :, mask_idxs, :] = np.nan

    # update keypoints2d list
    updated_human_data_list = []
    for i in range(view_number):
        # import pdb; pdb.set_trace()
        update_kps = keypoints2d_np[i, ...]
        update_kps = update_kps[:, np.newaxis, ...]
        updated_keypoints2d = Keypoints(
                dtype='numpy',
                kps=update_kps,
                mask=update_kps[..., -1] > 0,
                convention='human_data')

        updated_human_data_list.append(updated_keypoints2d)

    return kps_thr, valid_num_cam, updated_human_data_list

def __try_keypoints_threshold__(keypoints2d_np: np.ndarray,
                                keypoints2d_mask: np.ndarray,
                                start: float = 0.95,
                                stride: float = -0.05,
                                lower_bound: float = 0.0):
    """Try the largest keypoints_threshold by loop, which can be represented as
    start+n*stride and makes number of valid views >= 2.

    Args:
        keypoints2d_np (np.ndarray):
            In shape [view_number, frame_number, keypoints_number, 3].
        keypoints2d_mask (np.ndarray):
            In shape [keypoints_number, ].
        start (float, optional):
            Init threshold, should be in (0, 1].
            Defaults to 0.95.
        stride (float, optional):
            Step of one loop, should be in (-start, 0).
            Defaults to -0.05.
        lower_bound (float, optional):
            Lower bound of threshold, should be in [0.0, start).
            Defaults to 0.0.

    Returns:
        float: The best keypoints_threshold.
    """
    assert start > 0 and start <= 1
    assert stride > (0 - start) and stride < 0
    keypoints2d_init = keypoints2d_np.copy()
    # keypoints2d_init[:, :, keypoints2d_mask == 0, :] = 1
    keypoints2d_init = keypoints2d_init[:, :, np.where(keypoints2d_mask == 1), :]
    # import pdb; pdb.set_trace()
    keypoints_thr = start
    while True:
        pair_fail = False
        if keypoints_thr < lower_bound:
            keypoints_thr = lower_bound
            break
        valid_mask = keypoints2d_init[..., -1] >= keypoints_thr
        valid_num_array = np.sum(valid_mask, axis=0)
        if np.any(valid_num_array < 2):
            pair_fail = True
        if pair_fail:
            keypoints_thr += stride
        else:
            break
    return keypoints_thr, valid_num_array

def get_image_array(background_dir, view):
    frame_dir = os.path.join(background_dir, view)
    n_frame = 0
    if len(frame_dir) > 0:
        frame_list = os.listdir(frame_dir)
        n_frame = len(frame_list)
        frame_list.sort()
        abs_frame_list = [
            os.path.join(frame_dir, frame_name) for frame_name in frame_list
        ]
        image_list = [
            cv2.imread(filename=image_path) for image_path in abs_frame_list
        ]
        image_array = np.asarray(image_list)
    else:
        image_array = None
    return image_array, n_frame

def load_60_cam_kps2d(annot_dir, base_dir2d):
    human_data_list = []
    human_data_dir = base_dir2d
    
    annot = np.load(annot_dir, allow_pickle=True).item()
    human_data_key_list = sorted(list(annot['cams'].keys()))

    for human_data_key in human_data_key_list:
        tmp_human_data_path = os.path.join(
            human_data_dir,
            f'human_data_{human_data_key}.npz')

        human_data = HumanData.fromfile(tmp_human_data_path)
        
        # convert HumanData to Keypoints2D
        keypoints_src_mask = human_data['keypoints2d_mask']
        keypoints_src = human_data['keypoints2d'][..., :3]
        n_frame = keypoints_src.shape[0]
        keypoints_src_mask = np.repeat(keypoints_src_mask[np.newaxis,:], n_frame, axis=0)
        keypoints_src_mask = np.expand_dims(keypoints_src_mask,axis=1)
        keypoints_src = np.expand_dims(keypoints_src,axis=1)
        
        keypoints2d = Keypoints(            
            dtype='numpy',
            kps=keypoints_src,
            mask=keypoints_src_mask,
            convention='human_data')
        
        human_data_list.append(keypoints2d)
    assert len(human_data_list) >= 2, 'HumanData fewer than 2!'

    # load Camera Parameters

    cam_parameters_key_list = sorted(list(annot['cams'].keys()))
    camera_parameter_list = []
    for cam_key in cam_parameters_key_list:
        camera_parameter = FisheyeCameraParameter(name=cam_key)
        K, R, T, dist_coeff_k, dist_coeff_p, dist_coeff_dict = \
            get_camera_param_from_annots(annot_dir, cam_key)
        camera_parameter.set_KRT(K, R, T)
        camera_parameter.set_dist_coeff(dist_coeff_k,dist_coeff_p)
        camera_parameter.inverse_extrinsic()
        camera_parameter_list.append(camera_parameter)

    assert len(camera_parameter_list) == len(human_data_list),\
        'numbers of cameras and HumanData do not match'

    return camera_parameter_list,human_data_list

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

sort = False
is_sort = 'sorted' if sort else 'unsorted'

len_sets = []
mpjpe_sets = []
smpl_repj_sets = []
kps_repj_sets = []
updated_kps_repj_sets = []

now = datetime.now()
file_header = now.strftime("%y%m%d%H%M%S")   
log_file = f'{log_root}/{file_header}_eval_log_{is_sort}.csv'
print("log: ",log_file)

logger = open(log_file, 'w')
writer = csv.writer(logger)
header_1 = [f'(view {is_sort})_set_name', 'date', 'person', 'gender', 'action', 'full_mpjpe_mm', 'body_mpjpe_mm', 'hand_mpjpe_mm','offset_hand_mpjpe_mm', 'reprojection_error_pixel', \
            'used_reprojection_error_pixel', 'thr', 'min_view', 'max_view', 'mean_view']
header_2 = [i for i in range(60)]
header_3 = [f'updated_{i}' for i in range(60)]
header = header_1 + header_2 + header_3
writer.writerow(header)

# load final train
train_name =set_list_tt[0]
print(train_name)
subset = os.path.join(subset_root_tt, f'{train_name}.txt')
file = open(subset,'r')
lines_raw = file.readlines()
train_list = [line.strip('\n') for line in lines_raw]

for set_name in set_list_tt:
    print(set_name)
    subset = os.path.join(subset_root, f'{set_name}.txt')
    file = open(subset,'r')
    lines_raw = file.readlines()
    eval_list = [line.strip('\n') for line in lines_raw]

    mpjpe_seqs = []
    smpl_repj_seqs = []
    kps_repj_seqs = []
    updated_kps_repj_seqs = []

    for sequence in tqdm(eval_list):
        # for successful rate
        if sequence not in train_list:
            continue
        items = sequence.split("/")
        date = items[0]
        person = items[1]
        action = items[2]
        gender = person.split("_")[1]
        # person = f'{items[1]}_{items[2]}'
        # action = f'{items[3]}_{items[4]}_{items[5]}'
        # base_dir = os.path.join(root, date, person, action)
        base_dir3d = os.path.join(root3d, sequence)
        base_dir2d = os.path.join(root2d, sequence, 'pose_2d')
        annot_dir = os.path.join(root2d, date, person, 'annots.npy')
        background_dir = os.path.join(root2d, date, person, action, 'image')
        joint_dir = os.path.join(base_dir3d, 'smplx_xrmocap', 'human_data_tri_smplx_adhoc.npz')
        keypoints3d_dir = os.path.join(base_dir3d, 'smplx_xrmocap', 'human_data_optimized_xr_keypoints3d.npz')
        # smplx_dir = os.path.join(base_dir, 'human_data_smplx.npz')
        # keypoints_dir = os.path.join(base_dir, 'human_data_optimized_')

        info_dict = dict(np.load(joint_dir))

        # Keypoints and convert to smplx
        keypoints = info_dict['keypoints3d'][:, np.newaxis, ...]
        joints = info_dict['joints'][:, np.newaxis, ...]
        mask = info_dict['keypoints3d_mask'][:, np.newaxis, ...]

        kps_keypoints3d = Keypoints(            
            dtype='numpy',
            kps=keypoints,
            mask=mask,
            convention='smplx')
        
        full_keypoints3d = Keypoints.fromfile(keypoints3d_dir)
        # import pdb; pdb.set_trace()

        joints_keypoints3d = Keypoints(            
            dtype='numpy',
            kps=joints,
            mask=mask,
            convention='smplx')

        # MPJPE in SMPLX convention
        assert joints_keypoints3d.get_convention() == 'smplx'
        assert kps_keypoints3d.get_convention() == 'smplx'
        full_mpjpe_value = calc_mpjpe(joints_keypoints3d, kps_keypoints3d)
        body_mpjpe_value = calc_mpjpe(joints_keypoints3d, kps_keypoints3d, part_name='body')
        hand_mpjpe_value = calc_mpjpe(joints_keypoints3d, kps_keypoints3d, part_name='hand')
        offset_hand_mpjpe_value = calc_mpjpe(joints_keypoints3d, kps_keypoints3d, part_name='hand', offset=True)
        mpjpe_value = offset_hand_mpjpe_value

        cam_list, kps2d_list = load_60_cam_kps2d(annot_dir, base_dir2d)
        kps_repj_value, kps_repj_60 = calc_repj(cam_list, kps2d_list, 
                                                full_keypoints3d, background_dir, sequence, sort)
        kps_thr, valid_views, updated_kps2d_list= calc_stats(kps2d_list)
        updated_kps_repj_value, updated_kps_repj_60 = calc_repj(cam_list, updated_kps2d_list, 
                                                                full_keypoints3d, background_dir, sequence, sort)
        # updated_kps_repj_value = -1

        mpjpe_seqs.append(mpjpe_value)
        kps_repj_seqs.append(kps_repj_value)
        updated_kps_repj_seqs.append(updated_kps_repj_value)
        
        row = [set_name, date, person, gender, action, full_mpjpe_value, body_mpjpe_value, hand_mpjpe_value, offset_hand_mpjpe_value ,kps_repj_value, \
                updated_kps_repj_value, kps_thr] + valid_views + kps_repj_60 + updated_kps_repj_60
        writer.writerow(row)

    mpjpe_seqs = np.array(mpjpe_seqs)
    kps_repj_seqs = np.array(kps_repj_value)
    updated_kps_repj_seqs = np.array(updated_kps_repj_seqs)

    len_sets.append(len(mpjpe_seqs))
    mpjpe_sets.append(mpjpe_seqs.mean()*1000) # mm
    kps_repj_sets.append(kps_repj_seqs.mean()) # pixle
    updated_kps_repj_sets.append(updated_kps_repj_seqs.mean()) # pixle

table = BeautifulTable(max_width=80)
table.columns.insert(0, set_list, header='subset_name')
table.columns.insert(1, len_sets, header='#seq')
table.columns.insert(2, mpjpe_sets, header='MPJPE(mm)')
table.columns.insert(3, kps_repj_sets, header='kps3d reprojection error (pixel)')
table.columns.insert(4, updated_kps_repj_sets, header='used kps3d reprojection error (pixel)')

print(f'root: {root3d}')
print(table)

logger.close()
    
