import numpy as np
import torch
import torch.nn as nn 
import os.path as osp
import smplx
import pickle
import cv2 
from functools import partial

# from hawor_models.utils.rotation import batch_rodrigues, compute_twist_rotation

def append_value(x: torch.Tensor, value: float, dim=-1):
    r"""
    Append a value to a tensor in a specific dimension. (torch)

    e.g. append_value(torch.zeros(3, 3, 3), 1, dim=1) will result in a tensor of shape [3, 4, 3] where the extra
         part of the original tensor are all 1.

    :param x: Tensor in any shape.
    :param value: The value to be appended to the tensor.
    :param dim: The dimension to be expanded.
    :return: Tensor in the same shape except for the expanded dimension which is 1 larger.
    """
    app = torch.ones_like(x.index_select(dim, torch.tensor([0], device=x.device))) * value
    x = torch.cat((x, app), dim=dim)
    return x

append_zero = partial(append_value, value=0)
append_one = partial(append_value, value=1)

def vector_cross_matrix(x: torch.Tensor):
    r"""
    Get the skew-symmetric matrix :math:`[v]_\times\in so(3)` for each vector3 `v`. (torch, batch)

    :param x: Tensor that can reshape to [batch_size, 3].
    :return: The skew-symmetric matrix in shape [batch_size, 3, 3].
    """
    x = x.view(-1, 3)
    zeros = torch.zeros(x.shape[0], device=x.device)
    return torch.stack((zeros, -x[:, 2], x[:, 1],
                        x[:, 2], zeros, -x[:, 0],
                        -x[:, 1], x[:, 0], zeros), dim=1).view(-1, 3, 3)

def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)

def axis_angle_to_rotation_matrix(a: torch.Tensor):
    r"""
    Turn axis-angles into rotation matrices. (torch, batch)

    :param a: Axis-angle tensor that can reshape to [batch_size, 3].
    :return: Rotation matrix of shape [batch_size, 3, 3].
    """
    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis) | torch.isinf(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * vector_cross_matrix(axis)
    return r

def rotation_matrix_to_axis_angle(r: torch.Tensor):
    r"""
    Turn rotation matrices into axis-angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    import cv2
    result = [cv2.Rodrigues(_)[0] for _ in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    result = torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)
    return result

def transformation_matrix(R: torch.Tensor, p: torch.Tensor):
    r"""
    Get the homogeneous transformation matrices. (torch, batch)

    Transformation matrix :math:`T_{sb} \in SE(3)` of shape [4, 4] can convert points or vectors from b frame
    to s frame: :math:`x_s = T_{sb}x_b`.

    :param R: The rotation of b frame expressed in s frame, R_sb, in shape [*, 3, 3].
    :param p: The position of b frame expressed in s frame, p_s, in shape [*, 3].
    :return: The transformation matrix, T_sb, in shape [*, 4, 4].
    """
    Rp = torch.cat((R, p.unsqueeze(-1)), dim=-1)
    OI = torch.cat((torch.zeros(list(Rp.shape[:-2]) + [1, 3], device=R.device),
                    torch.ones(list(Rp.shape[:-2]) + [1, 1], device=R.device)), dim=-1)
    T = torch.cat((Rp, OI), dim=-2)
    return T

def inverse_transformation_matrix(T: torch.Tensor):
    r"""
    Get the inverse of the input homogeneous transformation matrices. (torch, batch)

    :param T: The transformation matrix in shape [*, 4, 4].
    :return: Matrix inverse in shape [*, 4, 4].
    """
    R, p = decode_transformation_matrix(T)
    invR = R.transpose(-1, -2)
    invp = -torch.matmul(invR, p.unsqueeze(-1)).squeeze(-1)
    invT = transformation_matrix(invR, invp)
    return invT

def decode_transformation_matrix(T: torch.Tensor):
    r"""
    Decode rotations and positions from the input homogeneous transformation matrices. (torch, batch)

    :param T: The transformation matrix in shape [*, 4, 4].
    :return: Rotation and position, in shape [*, 3, 3] and [*, 3].
    """
    R = T[..., :3, :3].clone()
    p = T[..., :3, 3].clone()
    return R, p

def _forward_tree(x_local: torch.Tensor, parent, reduction_fn):
    r"""
    Multiply/Add matrices along the tree branches. x_local [N, J, *]. parent [J].
    """
    x_global = [x_local[:, 0]]
    for i in range(1, len(parent)):
        x_global.append(reduction_fn(x_global[parent[i]], x_local[:, i]))
    x_global = torch.stack(x_global, dim=1)
    return x_global

def _inverse_tree(x_global: torch.Tensor, parent, reduction_fn, inverse_fn):
    r"""
    Inversely multiply/add matrices along the tree branches. x_global [N, J, *]. parent [J].
    """
    x_local = [x_global[:, 0]]
    for i in range(1, len(parent)):
        x_local.append(reduction_fn(inverse_fn(x_global[:, parent[i]]), x_global[:, i]))
    x_local = torch.stack(x_local, dim=1)
    return x_local

class SMPLX:
    _instance = None

    def __new__(cls,config=None):
        """Ensures only one instance of SMPL exists."""
        if cls._instance is None:
            assert config is not None, 'SMPLX requires human model path'
            cls._instance = super(SMPLX, cls).__new__(cls)
            cls._instance._initialize(config)
        return cls._instance

    def _initialize(self, human_model_path):
        print("=== human_model_path", human_model_path, flush=True)
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
        # self.layer = {'neutral': smplx.create(human_model_path, 'smplx', gender='NEUTRAL', use_pca=False, use_face_contour=True, **self.layer_arg),
        #                 'male': smplx.create(human_model_path, 'smplx', gender='MALE', use_pca=False, use_face_contour=True, **self.layer_arg),
        #                 'female': smplx.create(human_model_path, 'smplx', gender='FEMALE', use_pca=False, use_face_contour=True, **self.layer_arg)
        #                 }
        self.layer = {'neutral': smplx.SMPLXLayer(human_model_path+'/smplx', gender='NEUTRAL', use_pca=False, use_face_contour=True),
                        'male': smplx.SMPLXLayer(human_model_path+'/smplx', gender='MALE', use_pca=False, use_face_contour=True),
                        'female': smplx.SMPLXLayer(human_model_path+'/smplx', gender='FEMALE', use_pca=False, use_face_contour=True)
                        }
        self.vertex_num = 10475
        self.face = self.layer['neutral'].faces
        self.shape_param_dim = 10
        self.expr_code_dim = 10
        with open(osp.join(human_model_path, 'smplx', 'SMPLX_to_J14.pkl'), 'rb') as f:
            self.j14_regressor = pickle.load(f, encoding='latin1')
        with open(osp.join(human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            self.hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.face_vertex_idx = np.load(osp.join(human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
        self.J_regressor = self.layer['neutral'].J_regressor.numpy()
        self.J_regressor_idx = {'pelvis': 0, 'lwrist': 20, 'rwrist': 21, 'neck': 12}
        self.orig_hand_regressor = self.make_hand_regressor()
        #self.orig_hand_regressor = {'left': self.lhandayer.J_regressor.numpy()[[20,37,38,39,25,26,27,28,29,30,34,35,36,31,32,33],:], 'right': self.layer.J_regressor.numpy()[[21,52,53,54,40,41,42,43,44,45,49,50,51,46,47,48],:]}

        # original SMPLX joint set
        self.orig_joint_num = 53 # 22 (body joints) + 30 (hand joints) + 1 (face jaw joint)
        self.orig_joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', # right hand joints
        'Jaw' # face jaw joint
        )
        self.orig_flip_pairs = \
        ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body joints
        (22,37), (23,38), (24,39), (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51) # hand joints
        )
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_part = \
        {'body': range(self.orig_joints_name.index('Pelvis'), self.orig_joints_name.index('R_Wrist')+1),
        'lhand': range(self.orig_joints_name.index('L_Index_1'), self.orig_joints_name.index('L_Thumb_3')+1),
        'rhand': range(self.orig_joints_name.index('R_Index_1'), self.orig_joints_name.index('R_Thumb_3')+1),
        'face': range(self.orig_joints_name.index('Jaw'), self.orig_joints_name.index('Jaw')+1)}

        # changed SMPLX joint set for the supervision
        self.joint_num = 137 # 25 (body joints) + 40 (hand joints) + 72 (face keypoints)
        self.joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',# body joints
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
         *['Face_' + str(i) for i in range(1,73)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
         )
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.lwrist_idx = self.joints_name.index('L_Wrist')
        self.rwrist_idx = self.joints_name.index('R_Wrist')
        self.neck_idx = self.joints_name.index('Neck')
        self.flip_pairs = \
        ( (1,2), (3,4), (5,6), (8,9), (10,11), (12,13), (14,17), (15,18), (16,19), (20,21), (22,23), # body joints
        (25,45), (26,46), (27,47), (28,48), (29,49), (30,50), (31,51), (32,52), (33,53), (34,54), (35,55), (36,56), (37,57), (38,58), (39,59), (40,60), (41,61), (42,62), (43,63), (44,64), # hand joints
        (67,68), # face eyeballs
        (69,78), (70,77), (71,76), (72,75), (73,74), # face eyebrow
        (83,87), (84,86), # face below nose
        (88,97), (89,96), (90,95), (91,94), (92,99), (93,98), # face eyes
        (100,106), (101,105), (102,104), (107,111), (108,110), # face mouth
        (112,116), (113,115), (117,119), # face lip
        (120,136), (121,135), (122,134), (123,133), (124,132), (125,131), (126,130), (127,129) # face contours
        )
        self.joint_idx = \
        (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body joints
        37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand joints
        52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand joints
        22,15, # jaw, head
        57,56, # eyeballs
        76,77,78,79,80,81,82,83,84,85, # eyebrow
        86,87,88,89, # nose
        90,91,92,93,94, # below nose
        95,96,97,98,99,100,101,102,103,104,105,106, # eyes
        107, # right mouth
        108,109,110,111,112, # upper mouth
        113, # left mouth
        114,115,116,117,118, # lower mouth
        119, # right lip
        120,121,122, # upper lip
        123, # left lip
        124,125,126, # lower lip
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour
        )
        self.joint_part = \
        {'body': range(self.joints_name.index('Pelvis'), self.joints_name.index('Nose')+1),
        'lhand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.joints_name.index('R_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'hand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'face': range(self.joints_name.index('Face_1'), self.joints_name.index('Face_72')+1)}
        
        # changed SMPLX joint set for PositionNet prediction
        self.pos_joint_num = 65 # 25 (body joints) + 40 (hand joints)
        self.pos_joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose', # body joints
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
         )
        self.pos_joint_part = \
        {'body': range(self.pos_joints_name.index('Pelvis'), self.pos_joints_name.index('Nose')+1),
        'lhand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.pos_joints_name.index('R_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1),
        'hand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1)}
        self.pos_joint_part['L_MCP'] = [self.pos_joints_name.index('L_Index_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Middle_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Ring_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Pinky_1') - len(self.pos_joint_part['body'])]
        self.pos_joint_part['R_MCP'] = [self.pos_joints_name.index('R_Index_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Middle_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Ring_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Pinky_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand'])]
        
        self.prev_elbow_joints = [None, None]
    
    
    @classmethod
    def get_instance(cls):
        """Retrieve the singleton instance of SMPL."""
        return cls()

    def make_hand_regressor(self):
        regressor = self.layer['neutral'].J_regressor.numpy()
        lhand_regressor = np.concatenate((regressor[[20,37,38,39],:],
                                            np.eye(self.vertex_num)[5361,None],
                                                regressor[[25,26,27],:],
                                                np.eye(self.vertex_num)[4933,None],
                                                regressor[[28,29,30],:],
                                                np.eye(self.vertex_num)[5058,None],
                                                regressor[[34,35,36],:],
                                                np.eye(self.vertex_num)[5169,None],
                                                regressor[[31,32,33],:],
                                                np.eye(self.vertex_num)[5286,None]))
        rhand_regressor = np.concatenate((regressor[[21,52,53,54],:],
                                            np.eye(self.vertex_num)[8079,None],
                                                regressor[[40,41,42],:],
                                                np.eye(self.vertex_num)[7669,None],
                                                regressor[[43,44,45],:],
                                                np.eye(self.vertex_num)[7794,None],
                                                regressor[[49,50,51],:],
                                                np.eye(self.vertex_num)[7905,None],
                                                regressor[[46,47,48],:],
                                                np.eye(self.vertex_num)[8022,None]))
        hand_regressor = {'left': lhand_regressor, 'right': rhand_regressor}
        return hand_regressor

    def reduce_joint_set(self, joint):
        new_joint = []
        for name in self.pos_joints_name:
            idx = self.joints_name.index(name)
            new_joint.append(joint[:,idx,:])
        new_joint = torch.stack(new_joint,1)
        return new_joint

    def vertices2joints(self, gender, betas):
        device = betas.device 
        
        _v_template = self.layer[gender].v_template.clone().to(device)
        _shapedirs = self.layer[gender].shapedirs.clone().to(device)
        _J_regressor = self.layer[gender].J_regressor.clone().to(device)

        v = torch.einsum('bl,mkl->bmk', [betas, _shapedirs]) + _v_template
        j = torch.einsum('bik,ji->bjk', [v, _J_regressor])
        return j 

    def get_kinematic_tree(self, gender, num_joints=-1):
        parents = self.layer[gender].parents.clone()
        if num_joints < 0 :
            num_joints = len(parents) 

        parents = parents[:num_joints]
        children = torch.ones_like(parents) * -1
        for i in range(1, num_joints):
            if children[parents[i]] == -1:
                # the first child
                children[parents[i]] = i
            elif children[parents[i]] >= 0:
                # already has a child
                children[parents[i]] = -2
            else:
                assert children[parents[i]] < 0
                children[parents[i]] -= 1
        return parents, children 

    def get_zero_pose_joint_and_vertex(self, gender, betas, expression=None):
        device = betas.device 
        
        _v_template = self.layer[gender].v_template.clone().to(device)
        _shapedirs = self.layer[gender].shapedirs.clone().to(device)
        _expdirs = self.layer[gender].exp_dirs.clone().to(device)
        _J_regressor = self.layer[gender].J_regressor.clone().to(device)

        if expression is not None:
            # print("expression is not None", expression.size(), flush=True)
            shape_components = torch.cat([betas, expression], dim=-1)
            shapedirs = torch.cat([_shapedirs, _expdirs], dim=-1)
        else:
            shape_components = betas.clone()
            shapedirs = _shapedirs.clone()
            
        v = torch.einsum('bl,mkl->bmk', [shape_components, shapedirs]) + _v_template
        j = torch.einsum('bik,ji->bjk', [v, _J_regressor])
        j, v = j - j[:, :1], v - j[:, :1]
        return j, v

    def bone_vector_to_joint_position(self, gender, bone_vec):
        _parent = self.layer[gender].parents.clone().to(bone_vec.device)
        bone_vec = bone_vec.view(bone_vec.shape[0], -1, 3)
        joint_pos = _forward_tree(bone_vec, _parent, torch.add)
        return joint_pos

    def joint_position_to_bone_vector(self, gender, joint_pos):
        _parent = self.layer[gender].parents.clone().to(joint_pos.device)
        joint_pos = joint_pos.view(joint_pos.shape[0], -1, 3)
        bone_vec = _inverse_tree(joint_pos, _parent, torch.add, torch.neg)
        return bone_vec
    
    def forward_kinematics_R(self, gender, R_local):
        _parent = self.layer[gender].parents.clone().to(R_local.device)
        R_local = R_local.view(R_local.shape[0], -1, 3, 3)
        R_global = _forward_tree(R_local, _parent, torch.bmm)
        return R_global

    def forward_kinematics_T(self, gender, T_local):
        _parent = self.layer[gender].parents.clone().to(T_local.device)
        T_local = T_local.view(T_local.shape[0], -1, 4, 4)
        T_global = _forward_tree(T_local, _parent, torch.bmm)
        return T_global

    def forward_kinematics(self, gender, full_pose, shape, expr=None, tran=None, calc_mesh=False):
        
        def add_tran(x):
            return x if tran is None else x + tran.view(-1, 1, 3)

        batch = full_pose.shape[0]
        device = full_pose.device
        full_pose = full_pose.view(batch, -1, 3)
        pose = axis_angle_to_rotation_matrix(full_pose).view(batch, -1, 3, 3)
        
        j, v = [_.expand(batch, -1, -1) for _ in self.get_zero_pose_joint_and_vertex(gender, shape, expr)]
        T_local = transformation_matrix(pose, self.joint_position_to_bone_vector(gender, j))
        T_global = self.forward_kinematics_T(gender, T_local)
        pose_global, joint_global = decode_transformation_matrix(T_global)
        if not calc_mesh:
            return pose_global, add_tran(joint_global)

        T_global[..., -1:] -= torch.matmul(T_global, append_zero(j, dim=-1).unsqueeze(-1))
        _skinning_weights = self.layer[gender].lbs_weights.clone().to(device)
        T_vertex = torch.tensordot(T_global, _skinning_weights, dims=([1], [1])).permute(0, 3, 1, 2)
        vertex_global = torch.matmul(T_vertex, append_one(v, dim=-1).unsqueeze(-1)).squeeze(-1)[..., :3]
        return pose_global, add_tran(joint_global), add_tran(vertex_global)    
        
    def get_lbs_verts(self, gender, full_pose, shape, expr=None, tran=None):
        batch = full_pose.shape[0]
        device = full_pose.device
        
        #### 
        # pose = full_pose.reshape(-1, 165)
        # _pose_mean = self.layer[gender].pose_mean.clone().to(device)
        # pose += _pose_mean
        pose = full_pose.view(batch, -1, 3)
        pose = axis_angle_to_rotation_matrix(pose).view(batch, -1, 3, 3)

        _v_template = self.layer[gender].v_template.clone().to(device)
        _shapedirs = self.layer[gender].shapedirs.clone().to(device)
        _expdirs = self.layer[gender].shapedirs.clone().to(device)
        _posedirs = self.layer[gender].posedirs.clone().to(device)
        _J_regressor = self.layer[gender].J_regressor.clone().to(device)
        _parents = self.layer[gender].parents.clone().to(device)
        _skinning_weights = self.layer[gender].lbs_weights.clone().to(device)

        from smplx.lbs import lbs 
        if expr is not None:
            shape_components = torch.cat([shape, expr], dim=-1)
            shapedirs = torch.cat([_shapedirs, _expdirs], dim=-1)
        else:
            shape_components = shape.clone()
            shapedirs = _shapedirs.clone()

        vertex_global, _ = lbs(shape_components, pose, _v_template, shapedirs, _posedirs, _J_regressor, _parents, _skinning_weights, pose2rot=False)
        if tran is not None:
            vertex_global += tran.unsqueeze(dim=1)
        return vertex_global



class SMPL:
    _instance = None

    def __new__(cls,config=None):
        """Ensures only one instance of SMPL exists."""
        if cls._instance is None:
            assert config is not None, 'SMPL requires human model path'
            cls._instance = super(SMPL, cls).__new__(cls)
            cls._instance._initialize(config)
        return cls._instance

    def _initialize(self, human_model_path):
        """Initialize the SMPL model instance."""
        self.layer_arg = {
            'create_body_pose': False,
            'create_betas': False,
            'create_global_orient': False,
            'create_transl': False
        }
        self.layer = {
            'neutral': smplx.create(human_model_path, 'smpl', gender='NEUTRAL', **self.layer_arg),
            'male': smplx.create(human_model_path, 'smpl', gender='MALE', **self.layer_arg),
            'female': smplx.create(human_model_path, 'smpl', gender='FEMALE', **self.layer_arg)
        }
        self.vertex_num = 6890
        self.face = self.layer['neutral'].faces
        self.shape_param_dim = 10
        self.vposer_code_dim = 32

        # Original SMPL joint set
        self.orig_joint_num = 24
        self.orig_joints_name = (
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2',
            'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar',
            'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
            'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
        )
        self.orig_flip_pairs = (
            (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17),
            (18, 19), (20, 21), (22, 23)
        )
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_regressor = self.layer['neutral'].J_regressor.numpy().astype(np.float32)

        self.joint_num = self.orig_joint_num
        self.joints_name = self.orig_joints_name
        self.flip_pairs = self.orig_flip_pairs
        self.root_joint_idx = self.orig_root_joint_idx
        self.joint_regressor = self.orig_joint_regressor

    @classmethod
    def get_instance(cls):
        """Retrieve the singleton instance of SMPL."""
        return cls()


class ParametricModel:
    r"""
    SMPL/MANO/SMPLH parametric model.
    """
    def __init__(self, official_model_file: str, use_pose_blendshape=False, device=torch.device('cpu')):
        r"""
        Init an SMPL/MANO/SMPLH parametric model.

        :param official_model_file: Path to the official model to be loaded.
        :param use_pose_blendshape: Whether to use the pose blendshape.
        :param device: torch.device, cpu or cuda.
        """
        with open(official_model_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self._J_regressor = torch.from_numpy(data['J_regressor'].toarray()).float().to(device)
        self._skinning_weights = torch.from_numpy(data['weights']).float().to(device)
        self._posedirs = torch.from_numpy(data['posedirs']).float().to(device)
        self._shapedirs = torch.from_numpy(np.array(data['shapedirs'])).float().to(device)
        self._v_template = torch.from_numpy(data['v_template']).float().to(device)
        self._J = torch.from_numpy(data['J']).float().to(device)
        self.face = data['f']
        self.parent = data['kintree_table'][0].tolist()
        self.parent[0] = None
        self.use_pose_blendshape = use_pose_blendshape

    def save_obj_mesh(self, vertex_position, file_name='a.obj'):
        r"""
        Export an obj mesh using the input vertex position.

        :param vertex_position: Vertex position in shape [num_vertex, 3].
        :param file_name: Output obj file name.
        """
        with open(file_name, 'w') as fp:
            for v in vertex_position:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.face + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def get_zero_pose_joint_and_vertex(self, shape: torch.Tensor = None):
        r"""
        Get the joint and vertex positions in zero pose. Root joint is aligned at zero.

        :param shape: Tensor for model shapes that can reshape to [batch_size, 10]. Use None for the mean(zero) shape.
        :return: Joint tensor in shape [batch_size, num_joint, 3] and vertex tensor in shape [batch_size, num_vertex, 3]
                 if shape is not None. Otherwise [num_joint, 3] and [num_vertex, 3] assuming the mean(zero) shape.
        """
        if shape is None:
            j, v = self._J - self._J[:1], self._v_template - self._J[:1]
        else:
            shape = shape.view(-1, 10)
            v = torch.tensordot(shape, self._shapedirs, dims=([1], [2])) + self._v_template
            j = torch.matmul(self._J_regressor, v)
            j, v = j - j[:, :1], v - j[:, :1]
        return j, v

    def bone_vector_to_joint_position(self, bone_vec: torch.Tensor):
        r"""
        Calculate joint positions in the base frame from bone vectors (position difference of child and parent joint)
        in the base frame. (torch, batch)

        Notes
        -----
        bone_vec[:, i] is the vector from parent[i] to i.

        Args
        -----
        :param bone_vec: Bone vector tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
        :return: Joint position, in shape [batch_size, num_joint, 3].
        """
        bone_vec = bone_vec.view(bone_vec.shape[0], -1, 3)
        joint_pos = _forward_tree(bone_vec, self.parent, torch.add)
        return joint_pos

    def joint_position_to_bone_vector(self, joint_pos: torch.Tensor):
        r"""
        Calculate bone vectors (position difference of child and parent joint) in the base frame from joint positions
        in the base frame. (torch, batch)

        Notes
        -----
        bone_vec[:, i] is the vector from parent[i] to i.

        Args
        -----
        :param joint_pos: Joint position tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
        :return: Bone vector, in shape [batch_size, num_joint, 3].
        """
        joint_pos = joint_pos.view(joint_pos.shape[0], -1, 3)
        bone_vec = _inverse_tree(joint_pos, self.parent, torch.add, torch.neg)
        return bone_vec

    def forward_kinematics_R(self, R_local: torch.Tensor):
        r"""
        :math:`R_global = FK(R_local)`

        Forward kinematics that computes the global rotation of each joint from local rotations. (torch, batch)

        Notes
        -----
        A joint's *local* rotation is expressed in its parent's frame.

        A joint's *global* rotation is expressed in the base (root's parent) frame.

        Args
        -----
        :param R_local: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 3, 3] (rotation matrices).
        :return: Joint global rotation, in shape [batch_size, num_joint, 3, 3].
        """
        R_local = R_local.view(R_local.shape[0], -1, 3, 3)
        R_global = _forward_tree(R_local, self.parent, torch.bmm)
        return R_global

    def inverse_kinematics_R(self, R_global: torch.Tensor):
        r"""
        :math:`R_local = IK(R_global)`

        Inverse kinematics that computes the local rotation of each joint from global rotations. (torch, batch)

        Notes
        -----
        A joint's *local* rotation is expressed in its parent's frame.

        A joint's *global* rotation is expressed in the base (root's parent) frame.

        Args
        -----
        :param R_global: Joint global rotation tensor in shape [batch_size, *] that can reshape to
                         [batch_size, num_joint, 3, 3] (rotation matrices).
        :return: Joint local rotation, in shape [batch_size, num_joint, 3, 3].
        """
        R_global = R_global.view(R_global.shape[0], -1, 3, 3)
        R_local = _inverse_tree(R_global, self.parent, torch.bmm, partial(torch.transpose, dim0=1, dim1=2))
        return R_local

    def forward_kinematics_T(self, T_local: torch.Tensor):
        r"""
        :math:`T_global = FK(T_local)`

        Forward kinematics that computes the global homogeneous transformation of each joint from
        local homogeneous transformations. (torch, batch)

        Notes
        -----
        A joint's *local* transformation is expressed in its parent's frame.

        A joint's *global* transformation is expressed in the base (root's parent) frame.

        Args
        -----
        :param T_local: Joint local transformation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
        :return: Joint global transformation matrix, in shape [batch_size, num_joint, 4, 4].
        """
        T_local = T_local.view(T_local.shape[0], -1, 4, 4)
        T_global = _forward_tree(T_local, self.parent, torch.bmm)
        return T_global

    def inverse_kinematics_T(self, T_global: torch.Tensor):
        r"""
        :math:`T_local = IK(T_global)`

        Inverse kinematics that computes the local homogeneous transformation of each joint from
        global homogeneous transformations. (torch, batch)

        Notes
        -----
        A joint's *local* transformation is expressed in its parent's frame.

        A joint's *global* transformation is expressed in the base (root's parent) frame.

        Args
        -----
        :param T_global: Joint global transformation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
        :return: Joint local transformation matrix, in shape [batch_size, num_joint, 4, 4].
        """
        T_global = T_global.view(T_global.shape[0], -1, 4, 4)
        T_local = _inverse_tree(T_global, self.parent, torch.bmm, inverse_transformation_matrix)
        return T_local

    def forward_kinematics(self, pose: torch.Tensor, shape: torch.Tensor = None, tran: torch.Tensor = None,
                           calc_mesh=False):
        r"""
        Forward kinematics that computes the global joint rotation, joint position, and additionally
        mesh vertex position from poses, shapes, and translations. (torch, batch)

        :param pose: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                     [batch_size, num_joint, 3, 3] (rotation matrices).
        :param shape: Tensor for model shapes that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran: Root position tensor in shape [batch_size, 3]. Use None for the zero positions.
        :param calc_mesh: Whether to calculate mesh vertex positions.
        :return: Joint global rotation in [batch_size, num_joint, 3, 3],
                 joint position in [batch_size, num_joint, 3],
                 and additionally mesh vertex position in [batch_size, num_vertex, 3] if calc_mesh is True.
        """
        def add_tran(x):
            return x if tran is None else x + tran.view(-1, 1, 3)

        pose = pose.view(pose.shape[0], -1, 3, 3)
        j, v = [_.expand(pose.shape[0], -1, -1) for _ in self.get_zero_pose_joint_and_vertex(shape)]
        T_local = transformation_matrix(pose, self.joint_position_to_bone_vector(j))
        T_global = self.forward_kinematics_T(T_local)
        pose_global, joint_global = decode_transformation_matrix(T_global)
        if not calc_mesh:
            return pose_global, add_tran(joint_global)

        T_global[..., -1:] -= torch.matmul(T_global, append_zero(j, dim=-1).unsqueeze(-1))
        T_vertex = torch.tensordot(T_global, self._skinning_weights, dims=([1], [1])).permute(0, 3, 1, 2)
        if self.use_pose_blendshape:
            r = (pose[:, 1:] - torch.eye(3, device=pose.device)).flatten(1)
            v = v + torch.tensordot(r, self._posedirs, dims=([1], [2]))
        vertex_global = torch.matmul(T_vertex, append_one(v, dim=-1).unsqueeze(-1)).squeeze(-1)[..., :3]
        return pose_global, add_tran(joint_global), add_tran(vertex_global)
