from typing import Union

import torch

from ..builder import LOSSES

try:
    from pytorch3d.structures import Meshes
    from mesh_grid import (insert_grid_surface, search_nearest_point,
                           search_inside_mesh, search_intersect)
    mesh_grid_availabe = True
except ModuleNotFoundError:
    mesh_grid_availabe = False


@LOSSES.register_module()
class PointCloudChamferLoss(torch.nn.Module):

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 sample_num: int = 500):
        """PointCloudChamferLoss between two pointclouds.

        Args:
            reduction (str, optional):
               Choose from ['none', 'mean', 'sum']. Defaults to 'mean'.
            loss_weight (float, optional):
                Weight of silhouette loss. Defaults to 1.0.
            sample_num (int, optional):
                How many vertices will be sampled in one item.
                Defaults to 500.
        """
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.sample_num = sample_num

    def forward(self,
                src_verts: torch.Tensor,
                dst_verts: torch.Tensor,
                loss_weight_override: float = None,
                reduction_override: torch.Tensor = None) -> torch.Tensor:
        """Forward function of PointCloudChamferLoss. Randomly sample
        self.sample_num points from both src_verts and dst_verts, compute the
        distance between close pairs. For each sampled point in src_verts, find
        the closest point in dst_verts, average the square euclid distances
        among those pairs, and vice versa.

        Args:
            src_verts (torch.Tensor):
                3D vertices of point cloud A.
                In shape (batch_size, v_num_a, 3).
            dst_verts (torch.Tensor):
                3D vertices of point cloud B.
                In shape (batch_size, v_num_b, 3).
            loss_weight_override (float, optional):
                Temporal weight for this forward.
                Defaults to None, using self.loss_weight.
            reduction_override (torch.Tensor, optional):
                Reduction method along batch dim for this forward.
                Defaults to None, using self.reduction.

        Returns:
            torch.Tensor: Loss value in torch.Tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        verts_dict = {'src': src_verts, 'dst': dst_verts}
        b_size = src_verts.shape[0]
        sampled_dict = {}
        for key, verts in verts_dict.items():
            sample_index = torch.zeros(
                size=(b_size, self.sample_num),
                dtype=torch.int64,
                device=verts.device)
            for b_index in range(b_size):
                batch_sample_index = torch.randperm(
                    verts.shape[1], device=verts.device)[:self.sample_num]
                sample_index[b_index, :] = \
                    batch_sample_index + b_index * verts.shape[1]
            sample_index = sample_index.view(-1)
            sampled_verts = torch.index_select(
                verts.view(-1, 3), dim=0,
                index=sample_index).view(b_size, self.sample_num, 3)
            sampled_dict[key] = sampled_verts
        dist_mat = torch.cdist(
            sampled_dict['src'], sampled_dict['dst'], p=2)**2
        chamfer_distance = dist_mat.min(dim=1)[0] + dist_mat.min(dim=2)[0]
        chamfer_loss = torch.mean(chamfer_distance, dim=-1)
        if reduction == 'mean':
            chamfer_loss = torch.mean(chamfer_loss, dim=0)
        elif reduction == 'sum':
            chamfer_loss = torch.sum(chamfer_distance, dim=0)
        chamfer_loss *= loss_weight
        return chamfer_loss


class SurfaceNearest(torch.autograd.Function):

    @staticmethod
    def forward(ctx: object, points: torch.Tensor, verts: torch.Tensor,
                faces: torch.Tensor, tri_num: torch.Tensor,
                tri_idx: torch.Tensor, num: torch.Tensor,
                min_max: torch.Tensor, step: torch.Tensor) -> list:
        """Find the points neareast to meshes.

        Args:
            ctx (object):
                ctx is a context object that can be used to stash information。
            points (torch.Tensor):
                Points to be searched. In shape [p_num, 3].
            verts (torch.Tensor):
                Vertices of the meshes.
            faces (torch.Tensor):
                Faces of the meshes.
            tri_num (torch.Tensor): Shape [cube_num, ].
            tri_idx (torch.Tensor): Shape [1, ].
            num (torch.Tensor): Number of cubes: [a, b, c, a*b*c]
            min_max (torch.Tensor): Min location of the grid and max vert.
            step (torch.Tensor): Edge len of the average cube

        Returns:
            list: nearest_points and nearest_faces
        """
        device = verts.device
        nearest_faces = torch.zeros(
            points.shape[-2], dtype=torch.int32).to(device)
        coeff = torch.zeros(points.shape, dtype=torch.float32).to(device)
        nearest_pts = torch.zeros_like(coeff)
        search_nearest_point(points, verts, faces, tri_num, tri_idx, num,
                             min_max, step, nearest_faces, nearest_pts, coeff)
        ctx.save_for_backward(points, verts, faces, nearest_faces, coeff)
        return nearest_pts, nearest_faces


class MeshGridSearcher:

    def __init__(self,
                 meshes: Meshes = None,
                 verts: torch.Tensor = None,
                 faces: torch.Tensor = None,
                 device: Union[torch.device, str] = 'cuda'):
        """Create voxel grid according to meshes for fast search.

        Args:
            meshes (Meshes, optional):
                Meshes of a single object.
                If None, both verts and faces are necessary.
                Defaults to None.
            verts (torch.Tensor, optional):
                Vertices of the meshes.
                When meshes is given, verts will be ignored anyway.
                Defaults to None.
            faces (torch.Tensor, optional):
                Faces of the meshes.
                When meshes are given, faces will be ignored anyway.
                Defaults to None.
            device (Union[torch.device, str], optional):
                Device of the searcher. Defaults to 'cuda'.
        """
        self.device = device
        self.verts = None
        self.faces = None
        self.height = None
        self.step = None
        self.min_max = None
        self.num = None
        self.tri_num = None
        self.tri_idx = None
        assert mesh_grid_availabe,\
            'Module \'mesh_grid\' not installed.'
        if meshes is not None:
            self.set_meshes(meshes)
        else:
            assert verts is not None
            assert faces is not None
            self.set_mesh_by_verts_faces(verts, faces)

    def set_meshes(self, meshes: Meshes):
        """Set attr according to Meshes instance. Get grid and insert.

        Args:
            meshes (Meshes):
                Meshes of a single object in a single scene.
        """
        verts = meshes.verts_padded()[0]
        faces = meshes.faces_padded()[0]
        self.set_mesh_by_verts_faces(verts, faces)

    def set_mesh_by_verts_faces(self, verts: torch.Tensor,
                                faces: torch.Tensor):
        """Set attr according to verts and faces. Get grid and insert.

        Args:
            verts (torch.Tensor):
                Vertices of the meshes. In shape [v_num, 3] or [1, v_num, 3].
            faces (torch.Tensor):
                Faces of the meshes. In shape [f_num, 3] or [1, f_num, 3].
        """
        verts = verts.to(self.device)
        faces = faces.to(self.device, dtype=torch.int32)
        self.verts = verts.view(-1, 3)
        self.faces = faces.view(-1, 3)
        min_vert, _ = torch.min(verts, 0)
        max_vert, _ = torch.max(verts, 0)
        bbox_3d = max_vert - min_vert
        self.height = bbox_3d[1]
        bbox_center = (max_vert + min_vert) / 2
        # Averaging volume of the bbox_3d to each vert
        # self.step: edge len of the average cube
        self.step = (torch.cumprod(bbox_3d, 0)[-1] / verts.shape[0])**(1. / 3.)
        # Get cube number along each edge
        cube_number = torch.max(
            torch.floor(bbox_3d / self.step), torch.zeros_like(bbox_3d)) + 1
        # location of the gird point closest to (0, 0, 0)
        grid_min = bbox_center - self.step * cube_number / 2
        # concat number of cubes along bbox edges and bbox volume
        # self.num: [a, b, c, a*b*c]
        self.num = torch.cat([cube_number,
                              torch.cumprod(cube_number, 0)[-1:]]).int()
        self.min_max = torch.cat([grid_min, max_vert])

        self.tri_num = torch.zeros(
            self.num[-1], dtype=torch.int32).to(self.device)
        self.tri_idx = torch.zeros(1, dtype=torch.int32).to(self.device)

        insert_grid_surface(self.verts, self.faces, self.min_max, self.num,
                            self.step, self.tri_num, self.tri_idx)

    def nearest_points(self, points: torch.Tensor):
        """Find the points neareast to meshes.

        Args:
            points (torch.Tensor): In shape [p_num, 3].

        Returns:
            list: nearest_points and nearest_faces
        """
        points = points.to(self.device)
        return SurfaceNearest.apply(points, self.verts, self.faces,
                                    self.tri_num, self.tri_idx, self.num,
                                    self.min_max, self.step)

    def inside_mesh(self, points: torch.Tensor):
        """Find the points inside the meshes.

        Args:
            points (torch.Tensor): In shape [p_num, 3].

        Returns:
            torch.Tensor: Whether the point is inside. In shape [p_num].
        """
        points = points.to(self.device)
        inside = torch.zeros(
            points.shape[-2], dtype=torch.float32).to(self.device)
        search_inside_mesh(points, self.verts, self.faces, self.tri_num,
                           self.tri_idx, self.num, self.min_max, self.step,
                           inside)
        return inside

    def intersects_any(self, origins, directions):
        origins = origins.to(self.device)
        directions = directions.to(self.device)
        intersect = torch.zeros(
            origins.shape[-2], dtype=torch.bool).to(self.device)
        search_intersect(origins, directions, self.verts, self.faces,
                         self.tri_num, self.tri_idx, self.num, self.min_max,
                         self.step, intersect)
        return intersect


@LOSSES.register_module()
class PointCloudMeshGridLoss(torch.nn.Module):

    def __init__(self, reduction: str = 'mean', loss_weight: float = 1.0):
        """PointCloudChamferLoss between two pointclouds.

        Args:
            reduction (str, optional):
               Choose from ['none', 'mean', 'sum']. Defaults to 'mean'.
            loss_weight (float, optional):
                Weight of silhouette loss. Defaults to 1.0.
        """
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                body_verts: torch.Tensor,
                mesh_grid_searcher_list: list = None,
                meshes: Meshes = None,
                loss_weight_override: float = None,
                reduction_override: torch.Tensor = None) -> torch.Tensor:
        """Forward function of PointCloudChamferLoss. Randomly sample
        self.sample_num points from both src_verts and dst_verts, compute the
        distance between close pairs.

        Args:
            body_verts (torch.Tensor):
                Vertices of the body_model.
                In shape [b_size, v_num, 3].
            mesh_grid_searcher_list (list, optional):
                A list of MeshGridSearcher instances.
                If None, They will be constructed from meshes.
                Defaults to None.
            meshes (Meshes, optional):
                Meshes of batch_size objects.
            loss_weight_override (float, optional):
                Temporal weight for this forward.
                Defaults to None, using self.loss_weight.
            reduction_override (torch.Tensor, optional):
                Reduction method along batch dim for this forward.
                Defaults to None, using self.reduction.

        Returns:
            torch.Tensor: Loss value in torch.Tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)
        assert mesh_grid_searcher_list is not None or meshes is not None, \
            'Either mesh_grid_searcher_list or meshes must be given.'
        if mesh_grid_searcher_list is not None:
            assert len(mesh_grid_searcher_list) == body_verts.shape[0], \
                'Batch_size is not aligned!'
        else:
            assert len(meshes) == body_verts.shape[0], \
                'Batch_size is not aligned!'
        losses_in_batch = torch.zeros(
            size=(body_verts.shape[0], ),
            dtype=body_verts.dtype,
            device=body_verts.device,
        )
        for b_index in range(body_verts.shape[0]):
            mesh_grid_searcher = mesh_grid_searcher_list[b_index] \
                if mesh_grid_searcher_list is not None \
                else MeshGridSearcher(
                    meshes=meshes[b_index],
                    device=body_verts.device)
            b_verts = body_verts[b_index:b_index + 1, ...]
            closest_mesh_point, closest_mesh_face = \
                mesh_grid_searcher.nearest_points(b_verts.view(-1, 3))
            closest_distance = torch.norm(
                b_verts.view(-1, 3) - closest_mesh_point.detach(), p=2)
            losses_in_batch[b_index] = \
                torch.mean(closest_distance) / mesh_grid_searcher.height
        if reduction == 'mean':
            mesh_grid_loss = torch.mean(losses_in_batch, dim=0)
        elif reduction == 'sum':
            mesh_grid_loss = torch.sum(losses_in_batch, dim=0)
        mesh_grid_loss *= loss_weight
        return mesh_grid_loss
