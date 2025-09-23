import math
import random
from typing import List, Tuple, Union

import mmcv
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Scale, Translate
from tqdm import tqdm

from zoehuman.core.cameras.builder import build_cameras
from zoehuman.core.cameras.camera_parameters import CameraParameter
from zoehuman.core.cameras.cameras import compute_orbit_cameras
from zoehuman.core.conventions.cameras.convert_convention import (  # noqa:E501
    convert_camera_matrix, convert_K_4x4_to_3x3,
)
from zoehuman.core.visualization.renderer.torch3d_renderer.builder import \
    build_renderer  # prevent yapf isort conflict
from zoehuman.models.builder import build_body_model
from zoehuman.utils.transforms import ee_to_aa, ee_to_rotmat


def get_smpl_T_meshes(
        device: Union[torch.device, str] = 'cpu',
        color: Tuple = (1, 1, 1),
) -> Meshes:
    """Build a smpl body model and set it to T-pose.

    Args:
        device (Union[torch.device, str], optional):
            Device for torch.Tensor. Defaults to 'cpu'.
        color (Tuple, optional):
            Color for faces, in range [0, 1].
            Defaults to (1, 1, 1).

    Returns:
        Meshes: Meshes of the smpl model.
    """
    if isinstance(device, str):
        device = torch.device(device)
    # build body model
    smplify_config = dict(mmcv.Config.fromfile('configs/smplify/smplify.py'))
    body_model_config = smplify_config['body_model']
    model_type = body_model_config.get('type', 'smpl')
    body_model_config.update(type=model_type.lower())
    body_model = build_body_model(body_model_config)
    # set T-pose
    if model_type.lower() == 'smpl':
        poses = torch.zeros(1, 72)
    elif model_type.lower() == 'smplx':
        poses = torch.zeros(1, 165)
    else:
        raise TypeError
    global_orient = ee_to_aa(torch.tensor([math.pi, 0, 0])).view(1, 3)
    poses[:, :3] = global_orient
    pose_dict = body_model.tensor2dict(poses)
    smpl_output = body_model(**pose_dict)
    verts = smpl_output['vertices']
    smpl_template = Meshes(
        verts=verts,
        faces=body_model.faces_tensor.view(1, -1, 3),
        textures=TexturesVertex(
            verts_features=torch.FloatTensor(color).view(1, 1, 3).repeat(
                1, verts.shape[-2], 1))).to(device)
    return smpl_template


def normalize_verts(
        verts: torch.Tensor,
        rescale_by_height: float = 1.70,
        rescale_by_ratio: float = None,
        translate_by_location: Tuple[float, float,
                                     float] = None) -> torch.Tensor:
    """Normalize human verts, rescale and translate.

    Args:
        verts (torch.Tensor):
            Vertices to be normalized.
        rescale_by_height (float, optional):
            Height of verts after norm. Defaults to 1.70.
            It will be suppressed if rescale_by_ratio is given.
        rescale_by_ratio (float, optional):
            Ratio for rescale. Defaults to None.
            If ratio is given, rescale_by_height will be suppressed.
        translate_by_location (Tuple[float, float, float], optional):
            Vector for translation. Defaults to None.

    Returns:
        torch.Tensor: Transformed vertices.
    """
    device = verts.device
    rotation = ee_to_rotmat(torch.Tensor([math.pi, math.pi,
                                          0]).view(1, 3)).permute(0, 2, 1)
    rotation = Rotate(rotation, device=device)
    verts = rotation.transform_points(verts)
    # If the body is not vertical, should introduce mmpose detection
    # make G point over support foot
    # if ratio not specified, rescale by height
    if rescale_by_ratio is None:
        verts_height = abs(torch.max(verts[..., 1]) - torch.min(verts[..., 1]))
        ratio = rescale_by_height / verts_height
    else:
        ratio = rescale_by_ratio
    scalation = Scale(x=ratio, y=ratio, z=ratio, device=device)
    verts = scalation.transform_points(verts)
    # if location not specified, translate by smpl T-pose
    if translate_by_location is None:
        smpl_template = get_smpl_T_meshes(device=device)
        offset = torch.mean(smpl_template.verts_padded().view(-1, 3),
                            0) - torch.mean(verts.view(-1, 3), 0)
        offset_x, _, offset_z = torch.unbind(offset, -1)
        offset_y = torch.max(smpl_template.verts_padded().view(
            -1, 3)[:, 1]) - torch.max(verts.view(-1, 3)[:, 1])
    else:
        offset_x, offset_y, offset_z = translate_by_location
    translation = Translate(x=offset_x, y=offset_y, z=offset_z, device=device)
    verts = translation.transform_points(verts)
    return verts


def load_normed_human_mesh_from_obj(
    obj_path: str,
    rescale_by_height: float = 1.70,
    rescale_by_ratio: float = None,
    translate_by_location: Tuple[float, float, float] = None,
    device: Union[torch.device, str] = 'cpu',
) -> Meshes:
    """Load an obj file for human.

    Args:
        obj_path (str):
            Path to the obj file.
        rescale_by_height (float, optional):
            Height of verts after norm. Defaults to 1.70.
            It will be suppressed if rescale_by_ratio is given.
        rescale_by_ratio (float, optional):
            Ratio for rescale. Defaults to None.
            If ratio is given, rescale_by_height will be suppressed.
        translate_by_location (Tuple[float, float, float], optional):
            Vector for translation. Defaults to None.
        device (Union[torch.device, str], optional):
            Device for the meshes. Defaults to 'cpu'.

    Returns:
        Meshes: meshes for the normalized human.
    """
    mesh_raw = load_raw_human_mesh_from_obj(obj_path=obj_path, device=device)

    verts_raw = mesh_raw.verts_padded()[0].to(device).clone()
    verts_normed = normalize_verts(
        verts=verts_raw,
        rescale_by_height=rescale_by_height,
        rescale_by_ratio=rescale_by_ratio,
        translate_by_location=translate_by_location)
    mesh_normed = mesh_raw.update_padded(verts_normed[None])
    return mesh_normed


def load_raw_human_mesh_from_obj(
    obj_path: str,
    device: Union[torch.device, str] = 'cpu',
) -> Meshes:
    """Load an obj file for human.

    Args:
        obj_path (str):
            Path to the obj file.
        device (Union[torch.device, str], optional):
            Device for the meshes. Defaults to 'cpu'.

    Returns:
        Meshes: meshes for the human in obj file.
    """
    mesh_raw = load_objs_as_meshes([
        obj_path,
    ], device=device)
    return mesh_raw


def generate_cameras_lookingat(center: torch.Tensor,
                               camera_number: int = 36,
                               distance: float = 2.5,
                               target_bbox: torch.Tensor = None):
    """Generate several cameras looking at center.

    Args:
        center (torch.Tensor):
            The point [x, y, z] that all cameras looking at.
        camera_number (int, optional):
            Number of generated cameras. Defaults to 36.
        distance (float, optional):
            Distance between center and each camera.
            The real distance will be a random value in distance*[0.8, 1.2].
            Defaults to 2.5.
        target_bbox (torch.Tensor, optional):
            In shape [8, 3]. If given, auto_distance is enabled,
            and the camera will search from distance to far enough,
            to make sure that every point of target_bbox can be seen.
            Defaults to None.

    Raises:
        NotImplementedError: auto_distance is True

    Returns:
        K, R, T: Camera parameters in torch.Tensor.
        K: [camera_number, 4, 4]
        R: [camera_number, 3, 3]
        T: [camera_number, 3]
    """
    Ks = []
    Rs = []
    Ts = []
    angle_step = 0.1 * (360 / camera_number)
    target_bbox = target_bbox.cpu() if target_bbox is not None else None
    auto_distance = False if target_bbox is None else True
    for i in range(camera_number):
        # preprare init camera parameters
        elev = random.uniform(-15, 15)
        dist_offset = random.uniform(-0.2 * distance, 0.2 * distance)
        dist_init = dist_offset + distance
        azim = 360 // camera_number * i + random.uniform(
            -4 * angle_step, 4 * angle_step)
        # search 100 steps if auto
        if auto_distance:
            bbox_edges = torch.max(target_bbox, 0)[0] -\
                torch.min(target_bbox, 0)[0]
            max_edge = torch.max(bbox_edges, dim=0)[0]
            dist_step = 0.2 * max_edge
            max_step = 100
        # if not auto, return at the first step
        else:
            dist_step = 0.2 * distance
            max_step = 1
        # preprare for interation
        dist = dist_init
        sight_result = False
        step_count = 0
        while sight_result is False and step_count < max_step:
            # although convention='opencv', its output is in pytorch convention
            intrinsic, rotation, translation = compute_orbit_cameras(
                convention='opencv',
                elev=elev,
                dist=dist,
                azim=azim,
                at=center.cpu())
            # test bbox in NDC space
            if auto_distance:
                camera = build_cameras(
                    dict(
                        type='PerspectiveCameras',
                        K=intrinsic,
                        R=rotation,
                        T=translation,
                        in_ndc=True,
                        convention='pytorch3d',
                    ))
                ndc_bbox = camera.transform_points_ndc(target_bbox.view(-1, 3))
                sight_result = validate_ndc_sight(ndc_bbox)
            dist += dist_step
            step_count += 1
        Ks.append(intrinsic)
        Rs.append(rotation)
        Ts.append(translation)
    Ks = torch.cat(Ks)
    Rs = torch.cat(Rs)
    Ts = torch.cat(Ts)
    return Ks, Rs, Ts


def get_camera_parameters(Ks: torch.Tensor, Rs: torch.Tensor, Ts: torch.Tensor,
                          H: int, W: int) -> List:
    """Get a list of CameraParameter from paramteres.

    Args:
        Ks (torch.Tensor): In shape (cam_num, 4, 4).
        Rs (torch.Tensor): In shape (cam_num, 3, 3).
        Ts (torch.Tensor): In shape (cam_num, 3).
        H (int): Height of the cameras.
        W (int): Width of the cameras.

    Returns:
        List: A list of CameraParameter, length cam_num.
    """
    Ks, Rs, Ts = convert_camera_matrix(
        convention_dst='opencv',
        convention_src='pytorch3d',
        is_perspective=True,
        K=Ks,
        R=Rs,
        T=Ts,
        in_ndc_src=True,
        in_ndc_dst=False,
        resolution_dst=[H, W])
    Ks = convert_K_4x4_to_3x3(Ks, is_perspective=True)
    Ks = Ks.detach().cpu().numpy()
    Rs = Rs.detach().cpu().numpy()
    Ts = Ts.detach().cpu().numpy()
    cam_param_list = []
    for cam_index in range(Ks.shape[0]):
        cam_param = CameraParameter(name=f'cam_{cam_index:03d}', H=H, W=W)
        cam_param.set_KRT(
            K_mat=Ks[cam_index], R_mat=Rs[cam_index], T_vec=Ts[cam_index])
        cam_param.inverse_extrinsics()
        cam_param_list.append(cam_param)
    return cam_param_list


def render_meshes(meshes: Meshes,
                  output_path: str,
                  K: torch.Tensor,
                  R: torch.Tensor,
                  T: torch.Tensor,
                  resolution: Union[Tuple, List] = [1600, 1600],
                  device: Union[torch.device, str] = 'cpu',
                  disable_tqdm: bool = False):
    """Render mesh to cameras represented by KRT.

    Args:
        meshes (Meshes):
            Meshes for rendering.
        output_path (str):
            Path to image directory.
        K (torch.Tensor): In shape [cam_num, 4, 4].
        R (torch.Tensor): In shape [cam_num, 3, 3].
        T (torch.Tensor): In shape [cam_num, 3].
        resolution (Union[Tuple, List], optional):
            [H, W]. Defaults to [1600, 1600].
        device (Union[torch.device, str], optional):
            Device for the meshes. Defaults to 'cpu'.
        disable_tqdm (bool, optional):
            Whether to disable the entire progressbar wrapper.
            Not available until render in mmhuman3d updated.
            Defaults to False.
    """
    # number of cameras
    cam_num = R.shape[0]
    cameras = build_cameras(
        dict(
            type='PerspectiveCameras',
            K=K,
            R=R,
            T=T,
            in_ndc=True,
            image_size=resolution,
            convention='pytorch3d',
        ))
    renderer_cfg = dict(
        type='mesh',
        device=device,
        resolution=resolution,
        output_path=output_path)
    torch.cuda.empty_cache()
    renderer = build_renderer(renderer_cfg)
    with torch.no_grad():
        meshes = meshes.to(device)
        for cam_index in tqdm(range(cam_num), disable=disable_tqdm):
            camera = cameras[cam_index].to(device)
            _ = renderer(meshes=meshes, cameras=camera, indexes=[cam_index])


def validate_ndc_sight(ndc_points: torch.Tensor) -> bool:
    """Validate whether all the points in NDC space can be seen by the camera.
    If every point satisfies that z > 0 and x, y in (-1, 1), return True.

    Args:
        ndc_points (torch.Tensor):
            Points 3d in shape [..., 3].
            Defaults to None.

    Returns:
        bool: Whether the points are in sight.
    """
    ndc_points = ndc_points.view(-1, 3)
    sight = (ndc_points[:, 2] > 0).all() and\
        (ndc_points[:, 0] > -1).all() and\
        (ndc_points[:, 0] < 1).all() and\
        (ndc_points[:, 1] > -1).all() and\
        (ndc_points[:, 1] < 1).all()
    return bool(sight)
