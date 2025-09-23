import cv2
import numpy as np

import pytorch3d.renderer
import pytorch3d.transforms
import torch
import pytorch3d

from pytorch3d.structures import Meshes
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    TexturesVertex,
    SoftPhongShader,
    PointLights,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.io import save_obj
from typing import Optional, Tuple
from tqdm import trange
import trimesh


class Render(torch.nn.Module):
    def __init__(self, cam_param, img_size, device="cuda") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.rot = RotateAxisAngle(180, axis="Z", device=device)
        self.cameras = PerspectiveCameras(
            focal_length=cam_param["focal"],
            principal_point=cam_param["princpt"],
            in_ndc=False,
            image_size=(img_size,),
            device=device,
        )
        raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=raster_settings
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        self.shader = SoftPhongShader(
            device=device, cameras=self.cameras, lights=lights
        )

    def forward_fragments(
        self, vertices: torch.Tensor, faces: torch.Tensor
    ) -> Fragments:
        vertices = self.rot.transform_points(vertices)
        texture = TexturesVertex(verts_features=torch.ones_like(vertices))
        mesh = Meshes(verts=vertices, faces=faces, textures=texture).to(
            device=vertices.device
        )
        return self.rasterizer(mesh), mesh

    def forward_depth(
        self, vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_fragments(vertices, faces)[0].zbuf[..., 0].clip(0, None)

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor):
        fragments, mesh = self.forward_fragments(vertices, faces)
        rendered = self.shader(fragments, mesh)
        return rendered, fragments

    def forward_img(
        self,
        img: np.ndarray,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        return_depth=False,
    ):
        rendered, fragments = self.forward(vertices, faces)
        # save_obj('hand_mesh.obj', vertices.cpu().numpy(), faces.cpu().numpy())
        depth = fragments.zbuf[..., 0]
        rgb = rendered[..., :3]
        valid_mask = (depth > 0)[..., None].cpu().numpy()
        img = rgb.mul(255).to(dtype=torch.uint8).cpu().numpy() * valid_mask + img * (
            1 - valid_mask
        )
        if return_depth:
            return img, depth
        else:
            return img


def save_pose_to_obj(vertices, faces, save_fn):
    vertices = vertices.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # Save the mesh to an OBJ file
    mesh.export(save_fn)

def norm_depth(x, bg=0.0, scale=True):  # absolute ver
    # return (x/x.amax(dim=(-1,-2), keepdim=True)).clip(0,None)
    mask = x <= 0
    x_ = torch.where(mask, torch.inf, x)
    vmin = x_.amin(dim=(-1, -2), keepdim=True)
    vmax = x.amax(dim=(-1, -2), keepdim=True)
    # vmin = x[~mask].min()
    # vmax = x.max()
    if scale:
        x = ((x - vmin) / (vmax - vmin)).clip(0, None)
    x = torch.where(mask, bg, x)
    return x


def norm_depth_by_mask(x, y, bg=0.0, scale=True):
    x_ = x.clone()
    x_[y <= 0] = torch.nan
    x_[x <= 0] = torch.nan
    x_ = x_.flatten(-2, -1)
    vmin = torch.nanquantile(x_, 0.005, dim=-1, keepdim=True)[..., None]
    vmax = torch.nanquantile(x_, 0.980, dim=-1, keepdim=True)[..., None]
    x = x.clone()
    m = x < vmin
    M = x > vmax
    if scale:
        x = (x - vmin) / (vmax - vmin)
    x[m] = bg
    x[M] = bg
    return x


def fit_center(points: torch.Tensor):
    points_center = points.mean(0, keepdim=True)
    points -= points_center
    A = torch.einsum("...ji,...jk->...ik", points, points)
    b = 0.5 * (points * points.pow(2).sum(-1, True)).sum(0)
    est_center = torch.linalg.solve(A, b)
    est_center += points_center[0]
    return est_center


def fit_normal(points: torch.tensor):
    points_center = points.mean(0, keepdim=True)
    points -= points_center
    _, _, Vt = torch.linalg.svd(points)

    # The normal of the plane is the last row of Vt (corresponding to the smallest singular value)
    normal = Vt[-1]
    normal /= torch.linalg.norm(normal, 2, -1, keepdim=True)

    return normal



def make_sdf_depth(gt_depths: torch.Tensor, reproj_mask: torch.Tensor, bg_value=10.0):
    sdf_depth = []
    V = gt_depths.size(0) if gt_depths.dim() == 3 else 1
    reproj_mask_numpy = (reproj_mask).to(device="cpu", dtype=torch.uint8).numpy()
    kernel = np.ones((5, 5))
    for i in range(V):
        reproj_mask_numpy_i = cv2.morphologyEx(
            reproj_mask_numpy[i], cv2.MORPH_ERODE, kernel
        )
        res_pixel = cv2.distanceTransformWithLabels(
            1 - reproj_mask_numpy_i,
            distanceType=cv2.DIST_L2,
            maskSize=0,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        depth_image = gt_depths[i].cpu().numpy() * (reproj_mask_numpy_i)
        label_map = res_pixel[1]
        max_values = np.bincount(
            label_map.flatten(), weights=depth_image.flatten()
        )  # scatter_add
        result_image = max_values[label_map]
        sdf = res_pixel[0] / res_pixel[0].max()
        result_image = (1 - sdf) * result_image + sdf * bg_value
        result_image[reproj_mask_numpy[i]] = (
            gt_depths[i].cpu().numpy()[reproj_mask_numpy[i]]
        )
        sdf_depth.append(result_image)
    sdf_depth = np.stack(sdf_depth)
    return torch.tensor(sdf_depth, dtype=torch.float32, device="cuda")


def face_point(
    triangles: torch.Tensor, distances: torch.Tensor, closest_faces: torch.Tensor
) -> torch.Tensor:
    """
    Computes the squared distance of each triangular face in mesh to the closest
    point in pcd and averages across all faces in mesh.

    Returns
    -------
    avg_distance : Tensor
        A scalar tensor representing the average squared distance.
    """
    # Use BVHFunction to compute distances from points to the mesh surface
    # distances, _, closest_faces, _ = BVHFunction.apply(triangles, points)
    # distances: (B, N), squared distances from each point to the closest point on the mesh surface
    # closest_faces: (B, N), indices of the closest face for each point

    B, F, _, _ = triangles.shape
    B, N = distances.shape

    # Prepare for scatter operation
    batch_indices = (
        torch.arange(B, device=triangles.device).unsqueeze(1).expand(-1, N)
    )  # (B, N)
    # Compute global face indices
    global_face_indices = batch_indices * F + closest_faces  # (B, N)

    # Flattened distances and face indices
    flat_distances = distances.view(-1)  # (B*N,)
    flat_face_indices = global_face_indices.view(-1)  # (B*N,)

    # Total number of faces across all batches
    total_faces = B * F

    # Initialize min_distances with infinity
    min_distances = torch.full((total_faces,), float("inf"), device=triangles.device)

    # Use scatter_reduce to compute minimal distances per face
    min_distances = min_distances.scatter_reduce_(
        0, flat_face_indices, flat_distances, reduce="amin"
    )

    # Reshape back to (B, F)
    face_min_distances = min_distances.view(B, F)

    # Replace inf values with zeros (for faces that had no points mapped to them)
    face_min_distances = torch.where(
        face_min_distances == float("inf"), 0, face_min_distances
    )

    # Average across all faces
    avg_distance = face_min_distances.mean()
    return avg_distance


def transform_camera(
    vertices: torch.Tensor,
    bboxs: torch.Tensor,
    input_body_shape: torch.Tensor,
    tgt_princpt: Optional[torch.Tensor] = None,
    tgt_focal: Optional[torch.Tensor] = None,
):
    transformed_mesh = vertices.clone()
    centroid = transformed_mesh[..., 2].mean(-1)
    k = input_body_shape[1] / bboxs[:, 2]
    tz = centroid * (k - 1)
    transformed_mesh[..., 2] += tz[:, None]
    if tgt_princpt is not None and tgt_focal is not None:
        Z = transformed_mesh[..., 2:3]
        K = Z / tgt_focal[:, None, :]
        src_princpt = 0.5 * bboxs[..., 2:4] + bboxs[..., 0:2]
        d_princpt = K * (src_princpt - tgt_princpt)[:, None, :]
        transformed_mesh[..., :2] += d_princpt
    return transformed_mesh


def importance_sampling(data: torch.Tensor, weights: torch.Tensor, M: int):
    B, N, _ = data.shape

    # Sample indices without replacement
    indices = torch.multinomial(
        weights, num_samples=M, replacement=True
    )  # Shape: (B, M)

    # Create batch indices for advanced indexing
    batch_indices = torch.arange(B).unsqueeze(1).expand(-1, M)

    # Use advanced indexing to select samples
    sampled_data = data[batch_indices, indices]

    return sampled_data



def get_regressed_joint(vertices: torch.Tensor, J_regressor: torch.Tensor):
    joint = torch.einsum("bik,ji->bjk", [vertices, J_regressor])
    joint[..., 0] *= -1
    joint[..., 1] *= -1
    return joint


def calculate_visibility_score(visibility: torch.Tensor, J_regressor: torch.Tensor):
    V_full = torch.einsum(
        "bik,ji->bjk",
        [torch.ones_like(visibility, device="cuda", dtype=torch.float32), J_regressor],
    )  # full score
    V_ = torch.einsum(
        "bik,ji->bjk", [visibility.to(torch.float32), J_regressor]
    )  # regress vertice visibility to joint
    V_score = V_ / V_full
    idx_mask = torch.LongTensor([0, 1, 2, 4, 5, 7, 8, 10, 11])
    V_score[1, idx_mask] = 0
    # V_score = V_score.div(0.5).softmax(0) # norm along 'view' dim
    return V_score.div(V_score.sum(0, True)).squeeze(-1)  # norm along 'view' dim


def calculate_smplx_bone_length(joint3d: torch.Tensor):
    parent = torch.LongTensor(
        [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    )
    child = torch.arange(1, 22, dtype=torch.long)
    idx = torch.LongTensor([0, 2, 3])
    return torch.linalg.norm(
        joint3d[idx][:, parent] - joint3d[idx][:, child], 2, dim=-1, keepdims=True
    ).mean(0, keepdims=True)


import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


# def calculate_jacobian(fcn:callable, params: Tuple[torch.Tensor]|torch.Tensor, residual: torch.Tensor, batch_dim:Optional[int]=None):
#     if USE_FUNC:
#         pass
#     else:
#         jacobians = torch.autograd.functional.jacobian(fcn, params, create_graph=True, vectorize=True)

#         # Flatten parameters and Jacobians, and record shapes
#         flattened_params = []
#         param_shapes = []
#         for p in params:
#             flattened_params.append(p.reshape(-1))
#             param_shapes.append(p.shape)

#         # Reshape Jacobians and concatenate
#         flattened_jacobians = []
#         for jacobian_i, p_shape in zip(jacobians, param_shapes):
#             jacobian_i = jacobian_i.reshape(residual.numel(), -1)  # [N, p_i_size]
#             flattened_jacobians.append(jacobian_i)
#         total_jacobian = torch.cat(flattened_jacobians, dim=1)  # [N, total_param_size]
#     return total_jacobian, param_shapes


class HMGaussian(nn.Module):
    def __init__(
        self, num_gaussians: int = 1, mu: Optional[torch.Tensor] = None, D: int = 3
    ):
        super(HMGaussian, self).__init__()
        # D = 3  # Dimensionality
        self.D = D
        self.num_gaussians = num_gaussians

        # Initialize mean vector (mu) as a learnable parameter
        # Shape: [num_gaussians, D]
        if mu is not None:
            self.mu = nn.Parameter(mu, requires_grad=True)
        else:
            self.mu = nn.Parameter(torch.zeros(num_gaussians, D), requires_grad=True)

        # Initialize parameters for the covariance matrix
        # Shape: [num_gaussians, D, D]
        self.L_param = nn.Parameter(
            torch.eye(D).unsqueeze(0).repeat(num_gaussians, 1, 1), requires_grad=True
        )

        # Initialize bias term
        # Shape: [num_gaussians]
        # self.bias = nn.Parameter(torch.zeros(num_gaussians))

    def get_cholesky(self, L_param):
        L = torch.tril(L_param)  # Shape: [num_gaussians, D, D]
        diag_indices = torch.arange(self.D)
        L_diag = torch.exp(
            L[:, diag_indices, diag_indices]
        )  # Shape: [num_gaussians, D]
        L = L.clone()
        L[:, diag_indices, diag_indices] = L_diag
        return L

    def get_sig(self):
        L = self.get_cholesky(self.L_param).reshape(
            -1, self.num_gaussians, self.D, self.D
        )
        Sig = torch.einsum("...ij,...kj->...ik", [L, L])
        return Sig

    def forward(self, x):
        """
        Computes the log probability density (logits) of the input points under each Gaussian distribution.

        Args:
            x (torch.Tensor): Input tensor of shape [num_gaussians, N, D] or [N, D]

        Returns:
            log_prob (torch.Tensor): Log probabilities of shape [num_gaussians, N]
        """
        num_gaussians = self.num_gaussians

        # Ensure x has shape [num_gaussians, N, D]
        if x.dim() == 2:
            # x is of shape [N, D], expand to [num_gaussians, N, D]
            x = x.unsqueeze(0).expand(num_gaussians, -1, -1)
        elif x.dim() == 3:
            # x is already of shape [num_gaussians, N, D]
            assert x.shape[0] == num_gaussians, "Mismatch in number of gaussians"
        else:
            raise ValueError("Input x must be of shape [N, D] or [num_gaussians, N, D]")

        x = x.permute(1, 0, 2)

        # Ensure the covariance matrix is positive-definite via Cholesky decomposition
        L = self.get_cholesky(self.L_param)

        # Create a MultivariateNormal distribution for each Gaussian
        mvn = torch.distributions.MultivariateNormal(
            loc=self.mu,  # Shape: [num_gaussians, D]
            scale_tril=L,  # Shape: [num_gaussians, D, D]
        )

        # Compute the log probability density (logits)
        # We need to compute log_prob for each Gaussian and each data point
        log_prob = mvn.log_prob(x)  # Shape: [N, num_gaussians]

        return log_prob.permute(1, 0)

    def fit(self, x, y, num_epochs=1000, debug=False, lr=2e-2):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        if debug:
            pbar = trange(num_epochs)
        else:
            pbar = range(num_epochs)
        for epoch in pbar:
            optimizer.zero_grad()
            pred = self.forward(x).exp()
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0 and debug:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

    def fit_lm(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        u=1e-3,
        v=1.5,
        max_iter=30,
        mse_threshold=1e-8,
        verbose=False,
    ):
        last_update = 0
        last_mse = 0
        device = x.device
        num_gaussians = self.num_gaussians

        # Ensure x has shape [num_gaussians, N, D]
        if x.dim() == 2:
            # x is of shape [N, D], expand to [num_gaussians, N, D]
            x = x.unsqueeze(0).expand(num_gaussians, -1, -1)
        elif x.dim() == 3:
            # x is already of shape [num_gaussians, N, D]
            assert x.shape[0] == num_gaussians, "Mismatch in number of gaussians"
        else:
            raise ValueError("Input x must be of shape [N, D] or [num_gaussians, N, D]")

        x = x.permute(1, 0, 2)

        def fcn(mu, L_param):
            L = self.get_cholesky(L_param)
            mvn = torch.distributions.MultivariateNormal(
                loc=mu,  # Shape: [num_gaussians, D]
                scale_tril=L,  # Shape: [num_gaussians, D, D]
            )
            log_prob = mvn.log_prob(x)  # Shape: [N, num_gaussians]
            return (log_prob.permute(1, 0) - y).reshape(-1)

        # jacobian_fcn = vmap(jacrev(fcn, argnums=(0,1),), in_dims=((0,1,), (0,1)))

        params = (self.mu.detach().clone(), self.L_param.detach().clone())
        for i in range(max_iter):
            residual = fcn(*params)
            mse = torch.mean(residual**2)
            if abs(mse.item() - last_mse) < mse_threshold:
                if verbose:
                    print(f"Stopped at {i}, MSE: {mse.item()}")
                break  # Early stopping criterion met

            # Compute Jacobian
            jacobians = torch.autograd.functional.jacobian(
                fcn, params, create_graph=True, vectorize=True
            )

            # Flatten parameters and Jacobians, and record shapes
            flattened_params = []
            param_shapes = []
            for p in params:
                flattened_params.append(p.reshape(-1))
                param_shapes.append(p.shape)

            # Reshape Jacobians and concatenate
            flattened_jacobians = []
            for jacobian_i, p_shape in zip(jacobians, param_shapes):
                jacobian_i = jacobian_i.reshape(residual.numel(), -1)  # [N, p_i_size]
                flattened_jacobians.append(jacobian_i)
            total_jacobian = torch.cat(
                flattened_jacobians, dim=1
            )  # [N, total_param_size]

            # Compute J^T J and J^T residual
            jtj = total_jacobian.T @ total_jacobian
            jtj_damped = jtj + u * torch.eye(jtj.shape[0], device=device)
            jtr = total_jacobian.T @ residual

            # Solve for delta
            delta = torch.linalg.solve(jtj_damped, jtr)

            # Update parameters
            offset = 0
            new_params = []
            for idx, (p, p_shape) in enumerate(zip(params, param_shapes)):
                p_numel = p.numel()
                delta_i = delta[offset : offset + p_numel].reshape(p_shape)
                new_p = p - delta_i
                new_params.append(new_p.detach())
                offset += p_numel

            # Update damping parameter u
            update = last_mse - mse.item()
            if update > last_update and update > 0:
                u /= v
            else:
                u *= v

            last_update = update
            last_mse = mse.item()
            params = tuple(new_params)  # Update parameters for next iteration

            if verbose:
                print(f"Iteration {i}, MSE: {mse.item()}")
        self.mu.data = params[0]
        self.L_param.data = params[1]

    def get_maha_dist(self, points):
        diff = points - self.mu.detach()  # Shape: [N, D]
        # diff_T = diff.t()  # Shape: [D, N]
        L = self.get_cholesky(self.L_param).detach()

        # Transpose diff for solving
        diff_T = diff.unsqueeze(-1)  # Shape: [D, N, 1]

        # Solve L y = diff^T for y
        y = torch.linalg.solve_triangular(L, diff_T, upper=False).squeeze(
            -1
        )  # y has shape [D, N]
        # Compute Mahalanobis distances squared
        maha_dist_squared = torch.sum(y**2, dim=1)  # Shape: [N]

        # Compute Mahalanobis distances
        maha_dist = torch.sqrt(maha_dist_squared)  # Shape: [N]
        return maha_dist

from time import perf_counter
from functorch import jacfwd

class MVSmplx(torch.nn.Module):
    def __init__(
        self, smplx_layer, smplx_params, fit_delta=False, prev_params=None, is_right=False
    ) -> None:
        super().__init__()
        self.smplx_layer = smplx_layer
        self.is_right = is_right
        # self.lhand_pose = smplx_params["lhand_pose"]
        # self.rhand_pose = smplx_params["rhand_pose"]
        self.shape = smplx_params["shape"]
        # self.jaw_pose = smplx_params["jaw_pose"]
        # self.expr = smplx_params["expr"]

        self.fit_delta = fit_delta
        self.num_views = smplx_params["root_pose"].size(0)
        self.zero_pose = torch.zeros(
            (self.num_views, 3), dtype=torch.float32, device="cuda"
        )  # eye poses

        root_pose = smplx_params["root_pose"]
        body_pose = smplx_params["body_pose"]
        cam_trans =smplx_params["cam_trans"].detach().clone()
        if self.fit_delta:
            self.root_pose_ = prev_params["root_pose"].detach()
            self.body_pose_ = prev_params["body_pose"].detach()
            # root_pose = torch.zeros_like(root_pose)
            # body_pose = torch.zeros_like(body_pose)
            cam_trans = prev_params["cam_trans"].detach()
            root_pose = (root_pose - self.root_pose_).clone().detach()
            body_pose = (body_pose - self.body_pose_).clone().detach()
            # root_pose = (smplx_params['root_pose']-prev_params['prev_root_pose']).clone().detach()
            # body_pose = (smplx_params['body_pose']-prev_params['prev_body_pose']).clone().detach()
        else:
            self.root_pose_ = torch.zeros_like(root_pose)
            self.body_pose_ = torch.zeros_like(body_pose)

        self.root_pose = torch.nn.Parameter(root_pose, requires_grad=True)
        self.body_pose = torch.nn.Parameter(body_pose, requires_grad=True)

        xy_shift = cam_trans[:, 0:2]
        z_shift = cam_trans[:, 2:3]
        self.xy_shift = torch.nn.Parameter(xy_shift, requires_grad=True)
        self.z_shift = torch.nn.Parameter(z_shift, requires_grad=True)

    def get_body_pose(self):
        return self.body_pose + self.body_pose_

    def get_root_pose(self):
        return self.root_pose + self.root_pose_

    def get_cam_trans(self):
        return torch.cat((self.xy_shift, self.z_shift), -1).cuda()

    def forward_smplx(
        self,
        body_pose: torch.Tensor,
        global_orient: torch.Tensor,
        cam_trans: torch.Tensor,
    ):
        body_pose = body_pose.reshape(-1,3)
        global_orient = global_orient.reshape(-1,3)
        body_pose = pytorch3d.transforms.axis_angle_to_matrix(body_pose)
        global_orient = pytorch3d.transforms.axis_angle_to_matrix(global_orient)
        body_pose = body_pose.reshape(1,-1,3,3)
        global_orient = global_orient.reshape(1,-1,3,3)
        betas = self.shape.reshape(1, -1)
        output = self.smplx_layer(
            betas=betas,
            hand_pose=body_pose,
            global_orient=global_orient,
            pose2rot=False,
        )
        verts, joints = output.vertices, output.joints
        if not self.is_right:
            verts[..., 0] *= -1
            joints[..., 0] *= -1
        mesh_cam = verts + cam_trans[:, None, :]
        joint_cam = joints + cam_trans[:, None, :]
        return mesh_cam, joint_cam

    def forward(self, bboxs: torch.Tensor, input_body_shape, tgt_princpt, tgt_focal, img_size):
        # if self.fit_delta:
        body_pose = self.get_body_pose()
        global_orient = self.get_root_pose()
        cam_trans = self.get_cam_trans()
        # else:
        #     body_pose     = self.body_pose
        #     global_orient = self.root_pose
        mesh_cam, joint_cam = self.forward_smplx(body_pose, global_orient, cam_trans)
        # pred_mesh = transform_camera(
        #     mesh_cam, bboxs, input_body_shape, tgt_princpt, tgt_focal
        # )
        # pred_joint = transform_camera(
        #     joint_cam, bboxs, input_body_shape, tgt_princpt, tgt_focal
        # )
        return mesh_cam.clone(), joint_cam.clone()

    def solve_ik(
        self,
        tgt_joint: torch.Tensor,
        trans: dict,
        joint_mask: Optional[torch.Tensor] = None,
        u=1e-3,
        v=1.5,
        max_iter=30,
        mse_threshold=1e-8,
        verbose=False,
    ):
        # if joint_mask is None:
        #     joint_mask = torch.arange(144)
        def fcn_wrapper(body_pose, global_orient, cam_trans):
            _, joint_cam = self.forward_smplx(body_pose, global_orient, cam_trans)
            # pred_joint = transform_camera(vertices=joint_cam, **trans)
            pred_joint = joint_cam.clone()
            return (pred_joint[0, ...] - tgt_joint).reshape(-1)

        # Initialize parameters
        body_pose = self.get_body_pose()
        global_orient = self.get_root_pose()
        cam_trans = self.get_cam_trans()
        params = (body_pose, global_orient, cam_trans)

        last_update = 0
        last_mse = 0
        device = body_pose.device  # Assuming all tensors are on the same device

        for i in range(max_iter):
            # Compute residual
            residual = fcn_wrapper(*params)
            mse = torch.mean(residual**2)

            if abs(mse.item() - last_mse) < mse_threshold:
                if verbose:
                    print(f"Stopped at {i}, MSE: {mse.item()}")
                break  # Early stopping criterion met

            # Compute Jacobian
            # jacobians = torch.autograd.functional.jacobian(
            #     fcn_wrapper, params, create_graph=True, vectorize=True
            # )
            jacobians = jacfwd(fcn_wrapper, argnums=(0, 1, 2))(body_pose, global_orient, cam_trans)
            # Flatten parameters and Jacobians, and record shapes
            flattened_params = []
            param_shapes = []
            for p in params:
                flattened_params.append(p.reshape(-1))
                param_shapes.append(p.shape)
            # total_param_vector = torch.cat(flattened_params)
            # total_param_size = total_param_vector.numel()

            # Reshape Jacobians and concatenate
            flattened_jacobians = []
            for jacobian_i, p_shape in zip(jacobians, param_shapes):
                jacobian_i = jacobian_i.reshape(residual.numel(), -1)  # [N, p_i_size]
                flattened_jacobians.append(jacobian_i)
            total_jacobian = torch.cat(flattened_jacobians, dim=1)  # [N, total_param_size]
            # Compute J^T J and J^T residual
            jtj = total_jacobian.T @ total_jacobian
            jtj_damped = jtj + u * torch.eye(jtj.shape[0], device=device)
            jtr = total_jacobian.T @ residual

            # Solve for delta
            delta = torch.linalg.solve(jtj_damped, jtr)

            # Update parameters
            offset = 0
            new_params = []
            for idx, (p, p_shape) in enumerate(zip(params, param_shapes)):
                p_numel = p.numel()
                delta_i = delta[offset : offset + p_numel].reshape(p_shape)
                new_p = p - delta_i
                new_params.append(new_p.detach())
                offset += p_numel

            # Update damping parameter u
            update = last_mse - mse.item()
            if update > last_update and update > 0:
                u /= v
            else:
                u *= v

            last_update = update
            last_mse = mse.item()
            params = tuple(new_params)  # Update parameters for next iteration

            if verbose:
                print(f"Iteration {i}, MSE: {mse.item()}")
        wrong = mse.item() > 1e-4
        # Update the model's parameters after optimization
        self.body_pose.data = (params[0] - self.body_pose_)  # Adjust based on your initial setup
        self.root_pose.data = params[1] - self.root_pose_
        self.xy_shift.data = params[2][:, :2]
        self.z_shift.data = params[2][:, 2:]

        return wrong # Add return statements if needed


# from bvh_distance_queries.bvh_search_tree import BVH


class BodyPoseFit:
    def __init__(
        self,
        faces: torch.Tensor,
        rot: pytorch3d.transforms.RotateAxisAngle,
        R_i2j: torch.Tensor,
        T_i2j: torch.Tensor,
        base_view=3,
        fit_joint=False,
        joint_regressor: Optional[torch.Tensor] = None,
        w2c=None,
        fix_root=False,
        pen_isect=False,
        faces_segm=None,
        faces_parents=None,
        pcd_loss_w=20.0,
        joint_2d_w=1e-2,
        joint_3d_w=1e-2,
        joint_hm_w=0.0,
        col_loss_w=1e-2,
        pose_reg_w=0.0,
    ):
        self.faces = faces
        self.rot = rot
        self.R_i2j = R_i2j
        self.T_i2j = T_i2j
        self.base_view = base_view
        self.fit_joint = fit_joint
        self.J_regressor = joint_regressor
        self.w2c = w2c
        self.fix_root = fix_root
        self.pen_isect = pen_isect
        self.pcd_loss_w = pcd_loss_w
        self.joint_2d_w = joint_2d_w
        self.joint_3d_w = joint_3d_w
        self.joint_hm_w = joint_hm_w
        self.col_loss_w = col_loss_w
        self.pose_reg_w = pose_reg_w

        # Initialize reusable components


        # self.search_tree = BVH(queue_size=64)
        self.middle_bones_idx = torch.LongTensor([42, 43, 44, 57, 58, 59, 60, 61, 62])

        if self.fit_joint:
            self.joint_loss = torch.nn.MSELoss()
        if self.pose_reg_w > 0:
            self.reg_loss = torch.nn.L1Loss()

    def optimize(
        self,
        mv_smplx_obj: MVSmplx,
        transform_param: dict,
        gt_depth_pcd: Optional[torch.Tensor] = None,
        gt_joint_2d: Optional[torch.Tensor] = None,
        gt_joint_3d: Optional[torch.Tensor] = None,
        joint_hm_guassian: Optional[HMGaussian] = None,
        joint_mask: Optional[torch.Tensor] = None,
        epochs=100,
        early_return=True,
        verbose=False,
    ):
        if gt_depth_pcd is not None:
            M = 10000
            weights = compute_density_weights_knn(
                gt_depth_pcd.detach(), k=16, percentile=0.005
            )

        # Set up the optimizer
        if self.fix_root:
            optimizer = torch.optim.Adam(
                [
                    {"params": [mv_smplx_obj.xy_shift], "lr": 0},
                    {"params": [mv_smplx_obj.root_pose], "lr": 0},
                    {"params": [mv_smplx_obj.z_shift], "lr": 8e-3},
                    {"params": [mv_smplx_obj.body_pose], "lr": 8e-3},
                ]
            )
        else:                   
            optimizer = torch.optim.Adam(
                [
                    {"params": [mv_smplx_obj.xy_shift], "lr": 8e-3},
                    {"params": [mv_smplx_obj.root_pose], "lr": 5e-3},
                    {"params": [mv_smplx_obj.z_shift], "lr": 1e-2},
                    {"params": [mv_smplx_obj.body_pose], "lr": 2e-2},
                ]
            )
        pbar = trange(epochs) if verbose else range(epochs)
        for i in pbar:
            pred_mesh, pred_joint = mv_smplx_obj(**transform_param)
            factor = i / epochs
            joint_error = 0
            if self.fit_joint:
                if joint_mask is not None:
                    pred_joint = pred_joint[:, joint_mask]
                pred_joint_mv = (
                    torch.einsum(
                        "ni,bij->bnj",
                        pred_joint[0],
                        self.R_i2j[self.base_view].permute(0, 2, 1),
                    )
                    + self.T_i2j[self.base_view, :, None]
                )
                pred_joint_mv_rot = self.rot.transform_points(
                    pred_joint_mv
                ).contiguous()
                if gt_joint_2d is not None and self.joint_2d_w > 0:
                    pred_joint_mv_screen = self.w2c(pred_joint_mv_rot)
                    # pred_joint_mv_screen[..., :2] = transform_param['img_size'] - pred_joint_mv_screen[..., :2]
                    # import pdb; pdb.set_trace()
                    joint_error = joint_error + self.joint_2d_w * self.joint_loss(
                        pred_joint_mv_screen[..., :2], gt_joint_2d[..., :2]
                    )
                if gt_joint_3d is not None and self.joint_3d_w > 0:
                    joint_error = joint_error + self.joint_3d_w * self.joint_loss(
                        pred_joint_mv[0], gt_joint_3d
                    )
                if joint_hm_guassian is not None and self.joint_hm_w > 0:
                    joint_error = (
                        joint_error
                        + self.joint_hm_w
                        * joint_hm_guassian.get_maha_dist(pred_joint_mv[0]).sum()
                    )
            # joint_error = (1-factor+1e-1)*joint_error
            pred_mesh_0 = (
                torch.einsum(
                    "bni,ij->bnj",
                    pred_mesh[0:1],
                    self.R_i2j[self.base_view, 0].permute(1, 0),
                )
                + self.T_i2j[self.base_view, 0, None]
            )
            pred_mesh_0 = self.rot.transform_points(pred_mesh_0)
            triangles = pred_mesh_0[:, self.faces[0]]

            # if self.pose_reg_w and i == 2*epochs//5:
            #     init_pose = mv_smplx_obj.body_pose.clone().detach()
            # if i>2*epochs//5:
            pose_regulation = 0
            collision = 0
            if self.pen_isect:
                optimizer.param_groups[0]["lr"] = 0
                optimizer.param_groups[1]["lr"] = 0
                # joint_error = joint_error * 0.5
                with torch.no_grad():
                    collision_idxs = self.isec_search(triangles)
                    if self.filter_faces is not None:
                        collision_idxs = self.filter_faces(collision_idxs)
                collision = self.coll_joint_loss(
                    collision_idxs, pred_mesh_0, self.faces[0:1], self.J_regressor
                )
                if collision==0 and early_return: break
                collision = -self.col_loss_w * collision
                # collision = self.isec_loss(triangles, collision_idxs)
                # collision = self.col_loss_w * collision.mean()
                # if self.pose_reg_w > 0:
                #     pose_regulation = self.pose_reg_w * self.reg_loss(
                #         mv_smplx_obj.body_pose, init_pose
                #     )
                # else:
            # else:
            #     collision = pose_regulation = 0
            pcd_error = 0
            # if gt_depth_pcd is not None and self.pcd_loss_w > 0:
            #     # sampled_gt_pcd = importance_sampling(gt_depth_pcd.detach(), weights, M)
            #     # sampled_gt_pcd = gt_depth_pcd.detach()
            #     # coll_triangles = torch.cat((intr_triangles, recv_triangles), dim=1)
            #     # pred_mesh_0 = self.rot.transform_points(pred_mesh_0)
            #     # triangles = pred_mesh_0[:, self.faces[0]]
            #     point_face_dist, _, closest_faces, _ = self.search_tree(
            #         triangles, gt_depth_pcd.detach()
            #     )
            #     face_point_dist = face_point(triangles, point_face_dist, closest_faces)
            #     pcd_error = (face_point_dist + point_face_dist.mean()) * self.pcd_loss_w
            
            total_error = pcd_error + joint_error + collision + pose_regulation
            if verbose:
                print(
                    f"pcd:{pcd_error}, joint:{joint_error}, coll: {collision}"
                )
            if total_error == 0:
                continue
            optimizer.zero_grad()
            total_error.backward()
            # mv_smplx_obj.body_pose.grad[:, self.middle_bones_idx] = 0
            optimizer.step()
        
        wrong = total_error.item() > 1e-4
        pred_mesh, pred_joint = mv_smplx_obj(**transform_param)
        pred_mesh_mv = (
            torch.einsum(
                "ni,bij->bnj", pred_mesh[0], self.R_i2j[self.base_view].permute(0, 2, 1)
            )
            + self.T_i2j[self.base_view, :, None]
        )
        pred_joint_mv = (
            torch.einsum(
                "ni,bij->bnj",
                pred_joint[0],
                self.R_i2j[self.base_view].permute(0, 2, 1),
            )
            + self.T_i2j[self.base_view, :, None]
        )
        pred_joint_mv = self.rot.transform_points(pred_joint_mv).contiguous()
        pred_joint_mv_screen = self.w2c(pred_joint_mv)

        updated_params = {
            "root_pose": (mv_smplx_obj.root_pose + mv_smplx_obj.root_pose_)
            .detach()
            .clone(),
            "body_pose": (mv_smplx_obj.body_pose + mv_smplx_obj.body_pose_)
            .detach()
            .clone(),
            "cam_trans": torch.cat(
                (mv_smplx_obj.xy_shift.data, mv_smplx_obj.z_shift.data), -1
            )
            .detach()
            .clone(),
            'pred_joint_mv': pred_joint_mv.detach().clone(), #pred_joint_mv[0] if self.fit_joint else None,
            'pred_joint_mv_2d': pred_joint_mv_screen.detach().clone(),
        }

        return pred_mesh_mv, updated_params, wrong

    def optimize_lm(
        self,
        mv_smplx_obj: MVSmplx,
        transform_param: dict,
        gt_joint_3d: torch.Tensor,
        # gt_joint_2d: Optional[torch.Tensor] = None,
        # joint_2d_covar: Optional[torch.Tensor] = None,
        joint_mask: Optional[torch.Tensor] = None,
        u=1e-3,
        v=1.5,
        max_iter=30,
        mse_threshold=1e-8,
        verbose=False,
    ):
        def residual_fcn(body_pose, global_orient, cam_trans):
            mesh_cam, joint_cam = mv_smplx_obj.forward_smplx(
                body_pose, global_orient, cam_trans
            )
            pred_joint = transform_camera(vertices=joint_cam, **transform_param)
            if joint_mask is not None:
                pred_joint = pred_joint[:, joint_mask]
            pred_joint_0 = (
                torch.einsum(
                    "bni,ij->bnj",
                    pred_joint[0:1],
                    self.R_i2j[self.base_view, 0].transpose(-2, -1),
                )
                + self.T_i2j[self.base_view, 0, None]
            )
            residual_joint_3d = (pred_joint_0 - gt_joint_3d).reshape(-1)

            return residual_joint_3d, (residual_joint_3d,)

        def collision(body_pose, global_orient, cam_trans):
            mesh_cam, joint_cam = mv_smplx_obj.forward_smplx(
                body_pose, global_orient, cam_trans
            )
            pred_mesh = transform_camera(vertices=mesh_cam, **transform_param)
            triangles = pred_mesh[:, self.faces[0]]
            with torch.no_grad():
                collision_idxs = self.isec_search(triangles)
                if self.filter_faces is not None:
                    collision_idxs = self.filter_faces(collision_idxs)
            coll = self.isec_loss(triangles, collision_idxs).mean()
            return coll

        jacobian_fcn = jacfwd(residual_fcn, argnums=(0, 1, 2), has_aux=True)

        body_pose = mv_smplx_obj.get_body_pose()
        global_orient = mv_smplx_obj.get_root_pose()
        cam_trans = mv_smplx_obj.get_cam_trans()
        params = (body_pose, global_orient, cam_trans)

        param_shapes = []
        for p in params:
            param_shapes.append(p.shape)

        last_update = 0
        last_mse = 0
        device = body_pose.device
        pbar = trange(max_iter) if verbose else range(max_iter)
        for i in pbar:
            # residual = residual_fcn(*params)
            jacobians, (residual,) = jacobian_fcn(*params)

            params_ = []
            for p in params:
                p = p.detach()
                p.requires_grad = True
                if p.grad is not None:
                    p.grad.zero_()
                params_.append(p)
            coll_loss = collision(*params_)
            coll_loss.backward()
            coll_jac = torch.cat([p.grad.detach() for p in params_], dim=-1)
            mse = residual.pow(2).mean() + coll_loss
            if abs(mse.item() - last_mse) < mse_threshold:
                if verbose:
                    print(f"Stopped at {i}, MSE: {mse.item()}")
                break  # Early stopping criterion met
            residual = torch.cat((residual, coll_loss.unsqueeze(0)))
            total_jac = torch.cat(jacobians, dim=-1).squeeze(-2)
            total_jac = torch.cat((total_jac, coll_jac), dim=0)
            jtj = total_jac.T @ total_jac
            jtj_damped = jtj + u * torch.eye(jtj.shape[0], device=device)
            jtr = total_jac.T @ residual
            delta = torch.linalg.solve(jtj_damped, jtr)

            offset = 0
            new_params = []
            for idx, (p, p_shape) in enumerate(zip(params, param_shapes)):
                p_numel = p.numel()
                delta_i = delta[offset : offset + p_numel].reshape(p_shape)
                new_p = p - delta_i
                new_params.append(new_p.detach())
                offset += p_numel

            update = last_mse - mse.item()
            if update > last_update and update > 0:
                u /= v
            else:
                u *= v

            last_update = update
            last_mse = mse.item()
            params = tuple(new_params)

            if verbose:
                pbar.set_description(f"Iteration {i}, MSE: {mse.item()}")

        mv_smplx_obj.body_pose.data = params[0] - mv_smplx_obj.body_pose_
        mv_smplx_obj.root_pose.data = params[1] - mv_smplx_obj.root_pose_
        mv_smplx_obj.xy_shift.data = params[2][:, :2]
        mv_smplx_obj.z_shift.data = params[2][:, 2:]
        return mv_smplx_obj



class CollisonToJointLoss(torch.nn.Module):
    '''
    Collision triangles -> vertices -> joint.
    then push joint away.
    '''
    def __init__(self):
        super(CollisonToJointLoss, self).__init__()

    def forward(
        self,
        collision_idxs: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        joint_regressor: torch.Tensor,
    ):
        # B, N, J = vertices.size(0), vertices.size(1), joint_regressor.size(0)
        coll_idxs = collision_idxs[:, :, 0].ge(0).nonzero()
        if len(coll_idxs) >= 1:
            joint_regressor = joint_regressor.transpose(-1, -2)
            joints = torch.einsum(
                "...nd,nj->...jd", vertices, joint_regressor
            )  # BN3->BJ3
            joints = (
                joints.unsqueeze(-3)
                .sub(torch.zeros_like(joints).unsqueeze(-2))
                .transpose(-1, -3)
            )
            joints = joints.triu(0).detach() + joints.tril(-1) # On kinmatic tree, stop grad on root nodes, whose indices are smaller (upper dist matrix). By doing so, we only push arms or legs joints away instead of spine.
            joint_dists = joints.sub(joints.transpose(-1, -2)).pow(2).sum(-3).sqrt() # same as torch.cdist(joints, joints) # BJ3->BJJ
            with torch.no_grad():
                batch_idxs = coll_idxs[:, 0]
                batch_idxs_expanded = batch_idxs.unsqueeze(1).expand(-1, 3)
                batch_idxs_flat = batch_idxs_expanded.reshape(-1)

                intruder_faces = collision_idxs[coll_idxs[:, 0], coll_idxs[:, 1], 1]
                intr_vertex_indices = faces[batch_idxs, intruder_faces].reshape(-1)  # K
                intr_joint_score = joint_regressor[intr_vertex_indices]  # KJ

                receiver_faces = collision_idxs[coll_idxs[:, 0], coll_idxs[:, 1], 0]
                recv_vertex_indices = faces[batch_idxs, receiver_faces].reshape(-1)
                recv_joint_score = joint_regressor[recv_vertex_indices]

                score_matrix = torch.cdist(
                    intr_joint_score.unsqueeze(-1),
                    -recv_joint_score.unsqueeze(-1),
                    torch.inf,
                )  # kJJ
                mask = (
                    intr_joint_score.ne(0)
                    .unsqueeze(-1)
                    .logical_and(recv_joint_score.ne(0).unsqueeze(-2))
                )
            if ~mask.any():
                return torch.zeros(1, device=vertices.device)
            loss = joint_dists[batch_idxs_flat][mask] * score_matrix[mask]
            loss = loss.mean()
            return loss
        else:
            return torch.zeros(1, device=vertices.device)

def depth_to_camera_coords(joints2d, depth, K):
    """
    Convert a depth image to camera coordinate points.

    Parameters:
        depth (np.ndarray): Depth image of shape (H, W), where each element is the depth at that pixel.
        fx (float): Focal length in x-direction.
        fy (float): Focal length in y-direction.
        cx (float): Principal point x-coordinate.
        cy (float): Principal point y-coordinate.

    Returns:
        np.ndarray: Array of shape (H, W, 3) where each element is the [X, Y, Z] coordinate in the camera frame.
    """
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    u, v = joints2d[:, 0], joints2d[:, 1]
    d = depth[v.astype(int), u.astype(int)]
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    # Stack them into one array
    camera_coords = np.stack((X, Y, Z), axis=-1)
    return camera_coords

def load_mv(filepath):
    mv_smplx = torch.load(filepath)
    for k in mv_smplx.keys():
        mv_smplx[k] = mv_smplx[k].detach().cpu().numpy()
    return [mv_smplx['pred_joint_mv'], mv_smplx['pred_joint_mv_2d'], mv_smplx['body_pose'], mv_smplx['root_pose'], mv_smplx['cam_trans']]
