import os 
import torch
import numpy as np
import pickle
from typing import Optional
import smplx
from smplx.lbs import vertices2joints
from smplx.utils import MANOOutput, to_tensor
from smplx.vertex_ids import vertex_ids


class MANO(smplx.MANOLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        """
        Extension of the official MANO implementation to support more joints.
        Args:
            Same as MANOLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(MANO, self).__init__(*args, **kwargs)
        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        #2, 3, 5, 4, 1
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('extra_joints_idxs', to_tensor(list(vertex_ids['mano'].values()), dtype=torch.long))
        self.register_buffer('joint_map', torch.tensor(mano_to_openpose, dtype=torch.long))

    def forward(self, *args, **kwargs) -> MANOOutput:
        """
        Run forward pass. Same as MANO and also append an extra set of joints if joint_regressor_extra is specified.
        """
        mano_output = super(MANO, self).forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        mano_output.joints = joints
        return mano_output

class SMPLX_Wrapper(object):
    def __init__(self):
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
        # flat_hand_mean = True if model_type == 'mano' else False
        cur_cwd = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(cur_cwd))), "human_models/human_model_files")
        print("data_root", data_root, flush=True)
        self.layer = {'neutral': smplx.create(data_root, 'mano', gender='NEUTRAL', use_pca=False, use_face_contour=True, flat_hand_mean=True,**self.layer_arg),
                        'male': smplx.create(data_root, 'mano', gender='MALE', use_pca=False, use_face_contour=True, **self.layer_arg),
                        'female': smplx.create(data_root, 'mano', gender='FEMALE', use_pca=False, use_face_contour=True, **self.layer_arg)
                    }
        # self.vertex_num = 10475
        self.face = self.layer['neutral'].faces
        self.J_regressor = self.layer['neutral'].J_regressor.numpy()

# class SMPLX(object):
#     def __init__(self):
#         self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
#         self.layer = {'neutral': smplx.create(cfg.human_model_path, 'smplx', gender='NEUTRAL', use_pca=False, use_face_contour=True, **self.layer_arg),
#                         'male': smplx.create(cfg.human_model_path, 'smplx', gender='MALE', use_pca=False, use_face_contour=True, **self.layer_arg),
#                         'female': smplx.create(cfg.human_model_path, 'smplx', gender='FEMALE', use_pca=False, use_face_contour=True, **self.layer_arg)
#                         }
#         self.vertex_num = 10475
#         self.face = self.layer['neutral'].faces
#         self.shape_param_dim = 10

smpl_x = SMPLX_Wrapper()