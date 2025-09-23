from typing import Union

import numpy as np
import torch
from mmhuman3d.utils.mesh_utils import save_meshes_as_plys  # noqa:F401
from mmhuman3d.utils.path_utils import prepare_output_path
from pytorch3d.io import IO
from pytorch3d.io.obj_io import save_obj
from pytorch3d.renderer.mesh.textures import TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate

from zoehuman.utils.path_utils import check_path_suffix


def save_meshes_as_objs(obj_list, meshes: Meshes):
    if not isinstance(obj_list, list):
        obj_list = [obj_list]
    assert len(obj_list) >= len(meshes)
    if isinstance(meshes, Meshes):
        if isinstance(meshes.textures, TexturesVertex):
            for index, single_meshes in enumerate(meshes):
                IO().save_mesh(data=single_meshes, path=obj_list[index])
        elif isinstance(meshes.textures, TexturesUV):
            verts_uvs = meshes.textures.verts_uvs_padded()
            faces_uvs = meshes.textures.faces_uvs_padded()
            texture_maps = meshes.textures.maps_padded()
        else:
            verts_uvs = None
            faces_uvs = None
        verts = meshes.verts_padded()
        faces = meshes.faces_padded()
        for index in range(len(meshes)):
            prepare_output_path(
                obj_list[index], allowed_suffix=['.obj'], path_type='file')
            save_obj(
                obj_list[index],
                verts=verts[index],
                faces=faces[index],
                faces_uvs=faces_uvs[index],
                verts_uvs=verts_uvs[index],
                texture_map=texture_maps[index])
    else:
        raise NotImplementedError


def texture_uv2vc_t3d(meshes: Meshes):
    device = meshes.device
    vert_uv = meshes.textures.verts_uvs_padded()
    batch_size = vert_uv.shape[0]
    verts_features = []
    num_verts = meshes.verts_padded().shape[1]
    for index in range(batch_size):
        face_uv = vert_uv[index][meshes.textures.faces_uvs_padded()
                                 [index].view(-1)]
        img = meshes.textures._maps_padded[index]
        width, height, _ = img.shape
        face_uv = face_uv * torch.Tensor([width, height]).long().to(device)
        face_uv[:, 0] = torch.clip(face_uv[:, 0], 0, width - 1)
        face_uv[:, 1] = torch.clip(face_uv[:, 1], 0, height - 1)
        face_uv = face_uv.long()
        faces = meshes.faces_padded()
        verts_rgb = torch.zeros(1, num_verts, 3).to(device)
        verts_rgb[:, faces.view(-1)] = img[height - face_uv[:, 1], face_uv[:,
                                                                           0]]
        verts_features.append(verts_rgb)
    verts_features = torch.cat(verts_features)
    meshes = meshes.clone()
    meshes.textures = TexturesVertex(verts_features)
    return meshes


def axis_align_obb_rotation(
    points: Union[torch.FloatTensor,
                  np.ndarray], bbox: Union[torch.FloatTensor, np.ndarray]
) -> Union[torch.FloatTensor, np.ndarray]:
    """[summary]

    Args:
        points (Union[torch.FloatTensor, np.ndarray]): [description]
        bbox (Union[torch.FloatTensor, np.ndarray]): [description]

    Returns:
        Union[torch.FloatTensor, np.ndarray]: [description]
    """
    assert type(points) is type(
        bbox), 'Points and bbox should be the same type'
    device = torch.device('cpu')
    if isinstance(bbox, np.ndarray):
        bbox = torch.Tensor(bbox)
        points = torch.Tensor(points)
        data_type = 'numpy'
    else:
        data_type = 'tensor'
        device = points.device

    def norm(vec):
        vec = vec.view(-1, 3)
        # shape should be (n, 3), return the same shape normed vec
        return vec / torch.sqrt(vec[:, 0]**2 + vec[:, 1]**2 + vec[:, 2]**2)

    cross = np.corss if isinstance(bbox, np.ndarray) else torch.cross
    original_shape = points.shape
    points = points.reshape(-1, 3)
    bbox = bbox.reshape(8, 3)
    diff = bbox[0:1] - bbox
    length = diff[:, 0]**2 + diff[:, 1]**2 + diff[:, 2]**2
    max_bound = length.max() + 1
    length[0] = max_bound
    index1 = int(torch.argmin(length))
    length[index1] = max_bound
    index2 = int(torch.argmin(length))
    axis1 = norm(bbox[index1] - bbox[0])
    axis2 = norm(bbox[index2] - bbox[0])
    axis3 = cross(axis1, axis2)
    rotmat = torch.cat([axis1, axis2, axis3])
    arg = torch.argmax(abs(rotmat), 1)
    sort_index = [arg.tolist().index(i) for i in [0, 1, 2]]
    rotmat = rotmat[sort_index]
    sign = torch.sign(rotmat[torch.arange(3), torch.arange(3)])
    rotmat *= sign
    rotation = Rotate(rotmat.T, device=device)
    points = rotation.transform_points(points)
    points = points.reshape(original_shape)
    if data_type == 'numpy':
        points = points.numpy()
    else:
        points = points
    return points


def save_meshes(meshes: Meshes, save_path: str):
    """Save meshes to file.

    Args:
        meshes (Meshes):
            Meshes to save.
        save_path (str):
            The path for mesh file.

    Raises:
        TypeError: Type in save_path is not obj nor ply.
    """
    if check_path_suffix(save_path, ['.obj']):
        save_meshes_as_objs(obj_list=[save_path], meshes=meshes)
    elif check_path_suffix(save_path, ['.ply']):
        mesh_vc = texture_uv2vc_t3d(meshes)
        save_meshes_as_plys(paths=save_path, meshes=mesh_vc)
    else:
        raise TypeError
