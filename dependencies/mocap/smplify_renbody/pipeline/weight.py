def load_weight_shape(model):
    if model in ['smpl', 'smplh', 'smplx']:
        weight = {'s3d': 1., 'reg_shapes': 5e-3}
    elif model == 'mano':
        weight = {'s3d': 1e2, 'reg_shapes': 5e-5}
    else:
        raise NotImplementedError
    return weight


def load_weight_pose(model):
    if model == 'smpl':
        weight = {
            'k3d': 1.,
            'reg_poses_zero': 1e-2,
            'smooth_body': 5e0,
            'smooth_poses': 1e0,
            'reg_poses': 1e-3,
            'k2d': 1e-4
        }
    elif model == 'smplh':
        weight = {
            'k3d': 1.,
            'k3d_hand': 5.,
            'reg_poses_zero': 1e-2,
            'smooth_body': 5e-1,
            'smooth_poses': 1e-1,
            'smooth_hand': 1e-3,
            'reg_hand': 1e-4,
            'k2d': 1e-4
        }
    elif model == 'smplx':
        weight = {
            'k3d': 1.,
            'k3d_hand': 5.,
            'k3d_face': 2.,
            'reg_poses_zero': 1e-2,
            'smooth_body': 5e-1,
            'smooth_poses': 1e-1,
            'smooth_hand': 1e-3,
            'reg_hand': 1e-4,
            'reg_expr': 1e-2,
            'reg_head': 1e-2,
            'k2d': 1e-4
        }
    elif model == 'mano':
        weight = {
            'k3d': 1e2,
            'k2d': 2e-3,
            'reg_poses': 1e-3,
            'smooth_body': 1e2,
        }
        # weight = {
        #     'k3d': 1., 'k2d': 1e-4,
        #     'reg_poses': 1e-4, 'smooth_body': 0
        # }
    else:
        print(model)
        raise NotImplementedError
    return weight
