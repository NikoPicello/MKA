from typing import List, Tuple, Union

import torch
from mmhuman3d.models.registrants.smplify import OptimizableParameters
from mmhuman3d.models.registrants.smplify import SMPLify as SMPLify_mm
from mmhuman3d.models.registrants.smplify import build_optimizer
from pytorch3d.structures import Meshes

from zoehuman.data.datasets.pipelines.transforms import crop_resize_points
from zoehuman.models.builder import REGISTRANTS, build_loss
from zoehuman.models.losses.point_cloud_loss import MeshGridSearcher


@REGISTRANTS.register_module(force=True)
class SMPLify(SMPLify_mm):

    def __init__(
        self,
        body_model: Union[dict, torch.nn.Module],
        num_epochs: int = 20,
        camera: Union[dict, torch.nn.Module] = None,
        img_res: Union[Tuple[int], int] = 224,
        scale_factor: float = 1.0,
        stages: dict = None,
        optimizer: dict = None,
        keypoints2d_loss: dict = None,
        keypoints3d_loss: dict = None,
        silhouette_loss: dict = None,
        pointcloud_chamfer_loss: dict = None,
        pointcloud_meshgrid_loss: dict = None,
        shape_prior_loss: dict = None,
        joint_prior_loss: dict = None,
        smooth_loss: dict = None,
        pose_prior_loss: dict = None,
        use_one_betas_per_video: bool = False,
        ignore_keypoints: List[int] = None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose: bool = False,
    ) -> None:
        """
        Args:
            body_model: config or an object of body model.
            num_epochs: number of epochs of registration
            camera: config or an object of camera
            img_res: image resolution. If tuple, values are (width, height)
            scale_factor: input_metric * scale_factor = smpl_vert_metric
            stages: config of registration stages
            optimizer: config of optimizer
            keypoints2d_loss: config of keypoint 2D loss
            keypoints3d_loss: config of keypoint 3D loss
            silhouette_loss: config of silhouette loss.
            pointcloud_chamfer_loss: config of pointcloud chamfer loss
            pointcloud_meshgrid_loss: config of pointcloud mesh grid loss
            shape_prior_loss: config of shape prior loss.
                Used to prevent extreme shapes.
            joint_prior_loss: config of joint prior loss.
                Used to prevent large joint rotations.
            smooth_loss: config of smooth loss.
                Used to prevent jittering by temporal smoothing.
            pose_prior_loss: config of pose prior loss.
                Used to prevent
            use_one_betas_per_video: whether to use the same beta parameters
                for all frames in a single video sequence.
            ignore_keypoints: list of keypoint names to ignore in keypoint
                loss computation
            device: torch device
            verbose: whether to print individual losses during registration

        Returns:
            None
        """
        super().__init__(
            body_model=body_model,
            num_epochs=num_epochs,
            camera=camera,
            img_res=img_res,
            stages=stages,
            optimizer=optimizer,
            keypoints2d_loss=keypoints2d_loss,
            keypoints3d_loss=keypoints3d_loss,
            shape_prior_loss=shape_prior_loss,
            joint_prior_loss=joint_prior_loss,
            smooth_loss=smooth_loss,
            pose_prior_loss=pose_prior_loss,
            use_one_betas_per_video=use_one_betas_per_video,
            ignore_keypoints=ignore_keypoints,
            device=device,
            verbose=verbose,
        )
        self.silhouette_loss = build_loss(silhouette_loss)
        self.pointcloud_chamfer_loss = build_loss(pointcloud_chamfer_loss)
        self.pointcloud_meshgrid_loss = build_loss(pointcloud_meshgrid_loss)
        self.scale_factor = scale_factor

    def __call__(self,
                 keypoints2d: torch.Tensor = None,
                 keypoints2d_conf: torch.Tensor = None,
                 keypoints3d: torch.Tensor = None,
                 keypoints3d_conf: torch.Tensor = None,
                 silhouette_mask: torch.Tensor = None,
                 silhouette_transform: torch.Tensor = None,
                 human_mesh: Meshes = None,
                 init_global_orient: torch.Tensor = None,
                 init_transl: torch.Tensor = None,
                 init_body_pose: torch.Tensor = None,
                 init_betas: torch.Tensor = None,
                 return_verts: bool = False,
                 return_joints: bool = False,
                 return_full_pose: bool = False,
                 return_losses: bool = False) -> dict:
        """Run registration.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension
            Provide only keypoints2d or keypoints3d, not both.

        Args:
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            silhouette_mask:
                Human mask for silhouette extracting,
                in shape (B, m_size, m_size),  composed of 0 and 1.
                If None, disable silhouette loss.
            silhouette_transform:
                Parameters to transform points from camera space to mask space,
                in shape (B, 6).
                [:, :4] is bbox_xyxy and [:, 4:6] is mask size.
            human_mesh: Meshes of batch_size.
            init_global_orient: initial global_orient of shape (B, 3)
            init_transl: initial transl of shape (B, 3)
            init_body_pose: initial body_pose of shape (B, 69)
            init_betas: initial betas of shape (B, D)
            return_verts: whether to return vertices
            return_joints: whether to return joints
            return_full_pose: whether to return full pose
            return_losses: whether to return loss dict

        Returns:
            ret: a dictionary that includes body model parameters,
                and optional attributes such as vertices and joints
        """
        assert keypoints2d is not None or keypoints3d is not None, \
            'Neither of 2D nor 3D keypoints are provided.'
        assert not (keypoints2d is not None and keypoints3d is not None), \
            'Do not provide both 2D and 3D keypoints.'
        batch_size = keypoints2d.shape[0] if keypoints2d is not None \
            else keypoints3d.shape[0]

        global_orient = self._match_init_batch_size(
            init_global_orient, self.body_model.global_orient, batch_size)
        transl = self._match_init_batch_size(init_transl,
                                             self.body_model.transl,
                                             batch_size)
        body_pose = self._match_init_batch_size(init_body_pose,
                                                self.body_model.body_pose,
                                                batch_size)
        if init_betas is None and self.use_one_betas_per_video:
            betas = torch.zeros(1, self.body_model.betas.shape[-1]).to(
                self.device)
        else:
            betas = self._match_init_batch_size(init_betas,
                                                self.body_model.betas,
                                                batch_size)
        searcher_list = None
        if human_mesh is not None and \
                self.pointcloud_meshgrid_loss is not None:
            searcher_list = []
            for _, meshes in enumerate(human_mesh):
                searcher = MeshGridSearcher(meshes=meshes, device=self.device)
                searcher_list.append(searcher)

        for i in range(self.num_epochs):
            silhouette_mask_input = silhouette_mask
            human_mesh_input = human_mesh
            mesh_grid_searcher_list_input = searcher_list
            for stage_idx, stage_config in enumerate(self.stage_config):
                self._optimize_stage(
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    keypoints2d=keypoints2d,
                    keypoints2d_conf=keypoints2d_conf,
                    keypoints3d=keypoints3d,
                    keypoints3d_conf=keypoints3d_conf,
                    silhouette_mask=silhouette_mask_input,
                    silhouette_transform=silhouette_transform,
                    human_mesh=human_mesh_input,
                    mesh_grid_searchers=mesh_grid_searcher_list_input,
                    **stage_config,
                )

        # collate results
        ret = {
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'betas': betas
        }

        if return_verts or return_joints or \
                return_full_pose or return_losses:
            eval_ret = self.evaluate(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,
                keypoints2d=keypoints2d,
                keypoints2d_conf=keypoints2d_conf,
                keypoints3d=keypoints3d,
                keypoints3d_conf=keypoints3d_conf,
                silhouette_mask=silhouette_mask,
                silhouette_transform=silhouette_transform,
                return_verts=return_verts,
                return_full_pose=return_full_pose,
                return_joints=return_joints,
                reduction_override='none'  # sample-wise loss
            )

            if return_verts:
                ret['vertices'] = eval_ret['vertices']
            if return_joints:
                ret['joints'] = eval_ret['joints']
            if return_full_pose:
                ret['full_pose'] = eval_ret['full_pose']
            if return_losses:
                for k in eval_ret.keys():
                    if 'loss' in k:
                        ret[k] = eval_ret[k]

        for k, v in ret.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v.detach().clone()

        return ret

    def _optimize_stage(self,
                        betas: torch.Tensor,
                        body_pose: torch.Tensor,
                        global_orient: torch.Tensor,
                        transl: torch.Tensor,
                        fit_global_orient: bool = True,
                        fit_transl: bool = True,
                        fit_body_pose: bool = True,
                        fit_betas: bool = True,
                        keypoints2d: torch.Tensor = None,
                        keypoints2d_conf: torch.Tensor = None,
                        keypoints2d_weight: float = None,
                        keypoints3d: torch.Tensor = None,
                        keypoints3d_conf: torch.Tensor = None,
                        keypoints3d_weight: float = None,
                        silhouette_mask: torch.Tensor = None,
                        silhouette_transform: torch.Tensor = None,
                        silhouette_weight: float = None,
                        human_mesh: Meshes = None,
                        mesh_grid_searchers: list = None,
                        human_mesh_weight: float = None,
                        shape_prior_weight: float = None,
                        joint_prior_weight: float = None,
                        smooth_loss_weight: float = None,
                        pose_prior_weight: float = None,
                        joint_weights: dict = {},
                        num_iter: int = 1) -> None:
        """Optimize a stage of body model parameters according to
        configuration.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            betas: shape (B, D)
            body_pose: shape (B, 69)
            global_orient: shape (B, 3)
            transl: shape (B, 3)
            fit_global_orient: whether to optimize global_orient
            fit_transl: whether to optimize transl
            fit_body_pose: whether to optimize body_pose
            fit_betas: whether to optimize betas
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            silhouette_mask:
                Human mask for silhouette extracting,
                in shape (B, m_size, m_size),  composed of 0 and 1.
                If None, disable silhouette loss.
            silhouette_transform:
                Parameters to transform points from camera space to mask space,
                in shape (B, 6).
                [:, :4] is bbox_xyxy and [:, 4:6] is mask size.
            human_mesh: Meshes of batch_size.
            mesh_grid_searchers:
                A list of MeshGridSearcher, for pointcloud meshgrid loss.
                If mesh_grid_searchers is not None, pointcloud chamfer loss
                will be suppressed.
            human_mesh_weight: weight of point cloud mesh loss.
            silhouette_weight: weight of silhouette loss.
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            joint_weights: per joint weight of shape (K, )
            num_iter: number of iterations

        Returns:
            None
        """

        parameters = OptimizableParameters()
        parameters.set_param(fit_global_orient, global_orient)
        parameters.set_param(fit_transl, transl)
        parameters.set_param(fit_body_pose, body_pose)
        parameters.set_param(fit_betas, betas)

        optimizer = build_optimizer(parameters, self.optimizer)

        for iter_idx in range(num_iter):

            def closure():
                optimizer.zero_grad()
                betas_video = self._expand_betas(body_pose.shape[0], betas)

                loss_dict = self.evaluate(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_video,
                    transl=transl,
                    keypoints2d=keypoints2d,
                    keypoints2d_conf=keypoints2d_conf,
                    keypoints2d_weight=keypoints2d_weight,
                    keypoints3d=keypoints3d,
                    keypoints3d_conf=keypoints3d_conf,
                    keypoints3d_weight=keypoints3d_weight,
                    silhouette_mask=silhouette_mask,
                    silhouette_transform=silhouette_transform,
                    silhouette_weight=silhouette_weight,
                    human_mesh=human_mesh,
                    mesh_grid_searchers=mesh_grid_searchers,
                    human_mesh_weight=human_mesh_weight,
                    joint_prior_weight=joint_prior_weight,
                    shape_prior_weight=shape_prior_weight,
                    smooth_loss_weight=smooth_loss_weight,
                    pose_prior_weight=pose_prior_weight,
                    joint_weights=joint_weights)

                loss = loss_dict['total_loss']
                loss.backward()
                return loss

            optimizer.step(closure)

    def evaluate(
        self,
        betas: torch.Tensor = None,
        body_pose: torch.Tensor = None,
        global_orient: torch.Tensor = None,
        transl: torch.Tensor = None,
        keypoints2d: torch.Tensor = None,
        keypoints2d_conf: torch.Tensor = None,
        keypoints2d_weight: float = None,
        keypoints3d: torch.Tensor = None,
        keypoints3d_conf: torch.Tensor = None,
        keypoints3d_weight: float = None,
        silhouette_mask: torch.Tensor = None,
        silhouette_transform: torch.Tensor = None,
        silhouette_weight: float = None,
        human_mesh: Meshes = None,
        mesh_grid_searchers: list = None,
        human_mesh_weight: float = None,
        shape_prior_weight: float = None,
        joint_prior_weight: float = None,
        smooth_loss_weight: float = None,
        pose_prior_weight: float = None,
        joint_weights: dict = {},
        return_verts: bool = False,
        return_full_pose: bool = False,
        return_joints: bool = False,
        reduction_override: str = None,
    ) -> dict:
        """Evaluate fitted parameters through loss computation. This function
        serves two purposes: 1) internally, for loss backpropagation 2)
        externally, for fitting quality evaluation.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            betas: shape (B, D)
            body_pose: shape (B, 69)
            global_orient: shape (B, 3)
            transl: shape (B, 3)
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            silhouette_mask:
                Human mask for silhouette extracting,
                in shape (B, m_size, m_size),  composed of 0 and 1.
                If None, disable silhouette loss.
            silhouette_transform:
                Parameters to transform points from camera space to mask space,
                in shape (B, 6).
                [:, :4] is bbox_xyxy and [:, 4:6] is mask size.
            silhouette_weight: weight of silhouette loss.
            human_mesh: Meshes of batch_size.
            mesh_grid_searchers:
                A list of MeshGridSearcher, for pointcloud meshgrid loss.
                If mesh_grid_searchers is not None, pointcloud chamfer loss
                will be suppressed.
            human_mesh_weight: weight of point cloud mesh loss.
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            joint_weights: per joint weight of shape (K, )
            return_verts: whether to return vertices
            return_joints: whether to return joints
            return_full_pose: whether to return full pose
            reduction_override: reduction method, e.g., 'none', 'sum', 'mean'

        Returns:
            ret: a dictionary that includes body model parameters,
                and optional attributes such as vertices and joints
        """

        ret = {}

        body_model_return_verts = return_verts \
            if silhouette_mask is None and human_mesh is None \
            else True
        body_model_output = self.body_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            return_verts=body_model_return_verts,
            return_full_pose=return_full_pose)

        model_joints = body_model_output['joints']
        model_joint_mask = body_model_output['joint_mask']
        model_vertices = body_model_output.get('vertices', None)

        loss_dict = self._compute_loss(
            model_joints,
            model_joint_mask,
            model_vertices=model_vertices,
            keypoints2d=keypoints2d,
            keypoints2d_conf=keypoints2d_conf,
            keypoints2d_weight=keypoints2d_weight,
            keypoints3d=keypoints3d,
            keypoints3d_conf=keypoints3d_conf,
            keypoints3d_weight=keypoints3d_weight,
            silhouette_mask=silhouette_mask,
            silhouette_transform=silhouette_transform,
            silhouette_weight=silhouette_weight,
            human_mesh=human_mesh,
            mesh_grid_searchers=mesh_grid_searchers,
            human_mesh_weight=human_mesh_weight,
            joint_prior_weight=joint_prior_weight,
            shape_prior_weight=shape_prior_weight,
            smooth_loss_weight=smooth_loss_weight,
            pose_prior_weight=pose_prior_weight,
            joint_weights=joint_weights,
            reduction_override=reduction_override,
            body_pose=body_pose,
            betas=betas)
        ret.update(loss_dict)

        if return_verts:
            ret['vertices'] = body_model_output['vertices']
        if return_full_pose:
            ret['full_pose'] = body_model_output['full_pose']
        if return_joints:
            ret['joints'] = model_joints

        return ret

    def _compute_loss(self,
                      model_joints: torch.Tensor,
                      model_joint_conf: torch.Tensor,
                      model_vertices: torch.Tensor = None,
                      keypoints2d: torch.Tensor = None,
                      keypoints2d_conf: torch.Tensor = None,
                      keypoints2d_weight: float = None,
                      keypoints3d: torch.Tensor = None,
                      keypoints3d_conf: torch.Tensor = None,
                      keypoints3d_weight: float = None,
                      silhouette_mask: torch.Tensor = None,
                      silhouette_transform: torch.Tensor = None,
                      silhouette_weight: float = None,
                      human_mesh: Meshes = None,
                      mesh_grid_searchers: list = None,
                      human_mesh_weight: float = None,
                      shape_prior_weight: float = None,
                      joint_prior_weight: float = None,
                      smooth_loss_weight: float = None,
                      pose_prior_weight: float = None,
                      joint_weights: dict = {},
                      reduction_override: str = None,
                      body_pose: torch.Tensor = None,
                      betas: torch.Tensor = None):
        """Loss computation.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            model_joints: 3D joints regressed from body model of shape (B, K)
            model_joint_conf: 3D joint confidence of shape (B, K). It is
                normally all 1, except for zero-pads due to convert_kps in
                the SMPL wrapper.
            model_vertices: 3D vertices of body model, in shape (B, v_num, 3).
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            silhouette_mask:
                Human mask for silhouette extracting,
                in shape (B, m_size, m_size), composed of 0 and 1.
                If None, disable silhouette loss.
            silhouette_transform:
                Parameters to transform points from camera space to mask space,
                in shape (B, 6).
                [:, :4] is bbox_xyxy and [:, 4:6] is mask size.
            silhouette_weight: weight of silhouette loss.
            human_mesh: Meshes of batch_size.
            mesh_grid_searchers:
                A list of MeshGridSearcher, for pointcloud meshgrid loss.
                If mesh_grid_searchers is not None, pointcloud chamfer loss
                will be suppressed.
            human_mesh_weight: weight of point cloud mesh loss.
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            joint_weights: per joint weight of shape (K, )
            reduction_override: reduction method, e.g., 'none', 'sum', 'mean'
            body_pose: shape (B, 69), for loss computation
            betas: shape (B, D), for loss computation

        Returns:
            losses: a dict that contains all losses
        """
        scaled_keypoints2d = self.scale_factor * keypoints2d\
            if keypoints2d is not None else None
        scaled_keypoints3d = self.scale_factor * keypoints3d\
            if keypoints3d is not None else None
        # temporarily mute verbose when calling super()
        verbose_backup = self.verbose
        self.verbose = False
        losses = super()._compute_loss(
            model_joints=model_joints,
            model_joint_conf=model_joint_conf,
            keypoints2d=scaled_keypoints2d,
            keypoints2d_conf=keypoints2d_conf,
            keypoints2d_weight=keypoints2d_weight,
            keypoints3d=scaled_keypoints3d,
            keypoints3d_conf=keypoints3d_conf,
            keypoints3d_weight=keypoints3d_weight,
            joint_prior_weight=joint_prior_weight,
            shape_prior_weight=shape_prior_weight,
            smooth_loss_weight=smooth_loss_weight,
            pose_prior_weight=pose_prior_weight,
            joint_weights=joint_weights,
            reduction_override=reduction_override,
            body_pose=body_pose,
            betas=betas)

        if silhouette_mask is not None and\
                not self.__class__._skip_loss(
                    self.silhouette_loss, silhouette_weight
                ):
            selected_vertices = \
                model_vertices[:, ::4, :]  # select 1/4 for loss
            scaled_vertices = selected_vertices / self.scale_factor
            projected_vertices = self.camera.transform_points_screen(
                points=scaled_vertices)[..., :2]
            transformed_vertices = crop_resize_points(
                points2d=projected_vertices,
                bbox_xyxy=silhouette_transform[:, :4],
                dst_resolution=silhouette_transform[:, 4:6])
            # projected_vertices: (batch_size, vertices_number, 2)
            # masks: (batch_size, img_size, img_size)
            silhouette_loss = self.silhouette_loss.forward(
                transformed_vertices,
                silhouette_mask,
                loss_weight_override=silhouette_weight)
            losses['silhouette_loss'] = silhouette_loss
        if human_mesh is not None and not \
                (self.__class__._skip_loss(
                    self.pointcloud_chamfer_loss, human_mesh_weight) and
                 self.__class__._skip_loss(
                    self.pointcloud_meshgrid_loss, human_mesh_weight)
                 ):
            if mesh_grid_searchers is None:
                # mesh_vertices: (batch_size, verts_num, 3)
                mesh_vertices = human_mesh.verts_padded().to(
                    model_vertices.device)
                pointcloud_chamfer_loss = self.pointcloud_chamfer_loss(
                    src_verts=model_vertices,
                    dst_verts=mesh_vertices,
                    loss_weight_override=human_mesh_weight)
                losses['pointcloud_chamfer_loss'] = \
                    pointcloud_chamfer_loss
            else:
                pointcloud_meshgrid_loss = self.pointcloud_meshgrid_loss(
                    body_verts=model_vertices / self.scale_factor,
                    mesh_grid_searcher_list=mesh_grid_searchers,
                    loss_weight_override=human_mesh_weight)
                losses['pointcloud_meshgrid_loss'] = \
                    pointcloud_meshgrid_loss

        new_loss_names = [
            'silhouette_loss', 'pointcloud_chamfer_loss',
            'pointcloud_meshgrid_loss'
        ]

        total_loss = losses['total_loss']
        for loss_name, loss in losses.items():
            if loss_name in new_loss_names:
                if loss.ndim == 3:
                    total_loss = total_loss + loss.sum(dim=(2, 1))
                elif loss.ndim == 2:
                    total_loss = total_loss + loss.sum(dim=-1)
                else:
                    total_loss = total_loss + loss
        losses.pop('total_loss')
        losses['total_loss'] = total_loss

        self.verbose = verbose_backup
        if self.verbose:
            msg = ''
            for loss_name, loss in losses.items():
                msg += f'{loss_name}={loss.mean().item():.6f} '
            print(msg)
        return losses
