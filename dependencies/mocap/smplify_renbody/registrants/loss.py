import torch


def funcl1(x):
    return torch.sum(torch.abs(x**2))


def funcl2(x):
    return torch.sum(x**2)


def gmof(squared_res, sigma_squared):
    """Geman-McClure error function."""
    return (sigma_squared * squared_res) / (sigma_squared + squared_res)


class LossKeypoints3D:

    def __init__(self, keypoints3d, cfg, norm='l2') -> None:
        self.cfg = cfg
        keypoints3d = torch.Tensor(keypoints3d).to(cfg.device)
        self.nJoints = keypoints3d.shape[1]
        self.keypoints3d = keypoints3d[..., :3]
        self.conf = keypoints3d[..., 3:]
        self.nFrames = keypoints3d.shape[0]
        self.norm = norm

    def loss(self, diff_square):
        if self.norm == 'l2':
            loss_3d = funcl2(diff_square)
        elif self.norm == 'l1':
            loss_3d = funcl1(diff_square)
        elif self.norm == 'gm':
            loss_3d = torch.sum(gmof(diff_square**2, 0.04))
        else:
            raise NotImplementedError
        return loss_3d / self.nFrames

    def body(self, kpts_est, **kwargs):
        """distance of keypoints3d."""
        nJoints = min([kpts_est.shape[1], self.keypoints3d.shape[1], 25])
        diff_square = (
            kpts_est[:, :nJoints, :3] -
            self.keypoints3d[:, :nJoints, :3]) * self.conf[:, :nJoints]

        # # TODO: temp. bad case: bow
        # import numpy as np
        # save_path = '/home/caizhongang/github/zoehuman/data_temp/renbody/sensebee_datalist_83155/20220424/zhengwei_m/zhengwei_yf2_dz9/smplx_male/kp3d.npz'
        # np.savez(save_path,
        #          kpts_gt=self.keypoints3d[:, :nJoints, :3].detach().cpu().numpy(),
        #          kpts_est=kpts_est[:, :nJoints, :3].detach().cpu().numpy())
        return self.loss(diff_square)

    def hand(self, kpts_est, **kwargs):
        """distance of 3d hand keypoints."""
        diff_square = (
            kpts_est[:, 25:25 + 42, :3] -
            self.keypoints3d[:, 25:25 + 42, :3]) * self.conf[:, 25:25 + 42]
        return self.loss(diff_square)

    def face(self, kpts_est, **kwargs):
        """distance of 3d face keypoints."""
        diff_square = (kpts_est[:, 25 + 42:, :3] -
                       self.keypoints3d[:, 25 + 42:, :3]) * self.conf[:,
                                                                      25 + 42:]
        return self.loss(diff_square)

    def __str__(self) -> str:
        return 'Loss function for keypoints3D, norm = {}'.format(self.norm)


class LossRegPoses:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def reg_hand(self, poses, **kwargs):
        """regulizer for hand pose."""
        assert self.cfg.model in ['smplh', 'smplx']
        hand_poses = poses[:, 66:78]
        loss = funcl2(hand_poses)
        return loss / poses.shape[0]

    def reg_head(self, poses, **kwargs):
        """regulizer for head pose."""
        assert self.cfg.model in ['smplx']
        poses = poses[:, 78:]
        loss = funcl2(poses)
        return loss / poses.shape[0]

    def reg_expr(self, expression, **kwargs):
        """regulizer for expression."""
        assert self.cfg.model in ['smplh', 'smplx']
        return torch.sum(expression**2)

    def reg_body(self, poses, **kwargs):
        """regulizer for body poses."""
        if self.cfg.model in ['smplh', 'smplx']:
            poses = poses[:, :66]
        print('reg_body', poses.shape, poses.mean())
        loss = funcl2(poses)
        return loss / poses.shape[0]

    def __str__(self) -> str:
        return 'Loss function for Regulizer of Poses'


class LossRegPosesZero:

    def __init__(self, keypoints, cfg) -> None:
        model_type = cfg.model
        if keypoints.shape[-2] <= 15:
            use_feet = False
            use_head = False
        else:
            use_feet = keypoints[..., [19, 20, 21, 22, 23, 24], -1].sum() > 0.1
            use_head = keypoints[..., [15, 16, 17, 18], -1].sum() > 0.1
        if model_type == 'smpl':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14, 20, 21, 22, 23]
        elif model_type == 'smplh':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14]
        elif model_type == 'smplx':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14]
        else:
            raise NotImplementedError
        if not use_feet:
            SMPL_JOINT_ZERO_IDX.extend([7, 8])
        if not use_head:
            SMPL_JOINT_ZERO_IDX.extend([12, 15])
        SMPL_POSES_ZERO_IDX = [[j for j in range(3 * i, 3 * i + 3)]
                               for i in SMPL_JOINT_ZERO_IDX]
        SMPL_POSES_ZERO_IDX = sum(SMPL_POSES_ZERO_IDX, [])
        # SMPL_POSES_ZERO_IDX.extend([36, 37, 38, 45, 46, 47])
        self.idx = SMPL_POSES_ZERO_IDX

    def __call__(self, poses, **kwargs):
        """regulizer for zero joints."""
        return torch.sum(torch.abs(poses[:, self.idx])) / poses.shape[0]

    def __str__(self) -> str:
        return 'Loss function for Regulizer of Poses'


class LossSmoothBody:

    def __init__(self, cfg) -> None:
        self.norm = 'l2'

    def __call__(self, kpts_est, **kwargs):
        N_BODY = min(25, kpts_est.shape[1])
        assert kpts_est.shape[
            0] > 1, 'If you use smooth loss, it must be more than 1 frames'
        if self.norm == 'l2':
            loss = funcl2(kpts_est[:-1, :N_BODY] - kpts_est[1:, :N_BODY])
        else:
            loss = funcl1(kpts_est[:-1, :N_BODY] - kpts_est[1:, :N_BODY])
        return loss / kpts_est.shape[0]

    def __str__(self) -> str:
        return 'Loss function for Smooth of Body'


class LossSmoothBodyMean:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def smooth(self, kpts_est, **kwargs):
        """smooth body."""
        kpts_interp = kpts_est.clone().detach()
        kpts_interp[1:-1] = (kpts_interp[:-2] + kpts_interp[2:]) / 2
        loss = funcl2(kpts_est[1:-1] - kpts_interp[1:-1])
        return loss / (kpts_est.shape[0] - 2)

    def body(self, kpts_est, **kwargs):
        """smooth body."""
        return self.smooth(kpts_est[:, :25])

    def hand(self, kpts_est, **kwargs):
        """smooth body."""
        return self.smooth(kpts_est[:, 25:25 + 42])

    def __str__(self) -> str:
        return 'Loss function for Smooth of Body'


class LossSmoothPoses:

    def __init__(self, nViews, nFrames, cfg=None) -> None:
        self.nViews = nViews
        self.nFrames = nFrames
        self.norm = 'l2'
        self.cfg = cfg

    def _poses(self, poses):
        """smooth poses."""
        loss = 0
        for nv in range(self.nViews):
            poses_ = poses[nv * self.nFrames:(nv + 1) * self.nFrames, ]
            poses_interp = poses_.clone().detach()
            poses_interp[1:-1] = (poses_interp[1:-1] + poses_interp[:-2] +
                                  poses_interp[2:]) / 3
            loss += funcl2(poses_[1:-1] - poses_interp[1:-1])
        return loss / (self.nFrames - 2) / self.nViews

    def poses(self, poses, **kwargs):
        """smooth body poses."""
        # print('smooth body pose', poses.shape)
        if self.cfg.model in ['smplh', 'smplx']:
            poses = poses[:, :66]
        return self._poses(poses)

    def hands(self, poses, **kwargs):
        """smooth hand poses."""
        if self.cfg.model in ['smplh', 'smplx']:
            poses = poses[:, 66:66 + 12]
        else:
            raise NotImplementedError
        return self._poses(poses)

    def head(self, poses, **kwargs):
        """smooth head poses."""
        if self.cfg.model == 'smplx':
            poses = poses[:, 66 + 12:]
        else:
            raise NotImplementedError
        return self._poses(poses)

    def __str__(self) -> str:
        return 'Loss function for Smooth of Body'


class LossInit:

    def __init__(self, params, cfg) -> None:
        self.norm = 'l2'
        self.poses = torch.Tensor(params['poses']).to(cfg.device)
        self.shapes = torch.Tensor(params['shapes']).to(cfg.device)

    def init_poses(self, poses, **kwargs):
        """distance to poses_0."""
        if self.norm == 'l2':
            return torch.sum((poses - self.poses)**2) / poses.shape[0]

    def init_shapes(self, shapes, **kwargs):
        """distance to shapes_0."""
        if self.norm == 'l2':
            return torch.sum((shapes - self.shapes)**2) / shapes.shape[0]
