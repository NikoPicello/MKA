import os
# if 'PYOPENGL_PLATFORM' not in os.environ:
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import pyrender
import trimesh
import cv2
from yacs.config import CfgNode
from typing import List, Optional
from .geometry import perspective_projection
import matplotlib.pyplot as plt

def normalize_coordinates(box_center, img_size):
    """
    Normalize bounding box center to match the assumptions of get_camera_trans.

    Args:
        box_center (torch.Tensor): Bounding box center (N, 2) in pixel space.
        img_size (torch.Tensor): Image size (N, 2) in pixels.

    Returns:
        torch.Tensor: Normalized bounding box center (N, 2).
    """
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    norm_x = 2 * (box_center[:, 0] - img_w / 2) / img_w
    norm_y = 2 * (box_center[:, 1] - img_h / 2) / img_h
    return torch.stack([norm_x, norm_y], dim=-1)

def align_cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    """
    Normalize cam_crop_to_full to align with get_camera_trans.

    Args:
        cam_bbox (torch.Tensor): Camera parameters for cropped image (N, 3).
        box_center (torch.Tensor): Bounding box center in pixel space (N, 2).
        box_size (torch.Tensor): Bounding box size (N,).
        img_size (torch.Tensor): Full image size (N, 2).
        focal_length (float): Camera focal length.

    Returns:
        torch.Tensor: Camera translation aligned to normalized coordinates (N, 3).
    """
    # Normalize box center
    norm_box_center = normalize_coordinates(box_center, img_size)
    bs = box_size * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs  # Align depth scaling

    # Adjust normalized coordinates for tx, ty
    tx = norm_box_center[:, 0] + cam_bbox[:, 1]
    ty = norm_box_center[:, 1] + cam_bbox[:, 2]
    
    # Combine into full camera parameters
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses

def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)

def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    

def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

class Renderer:

    def __init__(self, cfg: CfgNode, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        self.cfg = cfg
        self.focal_length = cfg.EXTRA.FOCAL_LENGTH
        self.img_res = cfg.MODEL.IMAGE_SIZE

        # add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        
        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces
        self.faces_left = self.faces[:,[0,2,1]]

    def __call__(self,
                vertices: np.array,
                camera_translation: np.array,
                image: torch.Tensor,
                full_frame: bool = False,
                imgname: Optional[str] = None,
                side_view=False, rot_angle=90,
                mesh_base_color=(1.0, 1.0, 0.9),
                scene_bg_color=(0,0,0),
                return_rgba=False,
                ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """
        
        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        else:
            image = image.clone() * torch.tensor(self.cfg.MODEL.IMAGE_STD, device=image.device).reshape(3,1,1)
            image = image + torch.tensor(self.cfg.MODEL.IMAGE_MEAN, device=image.device).reshape(3,1,1)
            image = image.permute(1, 2, 0).cpu().numpy()

        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(*mesh_base_color, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        scene.add(camera, pose=camera_pose)


        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        if return_rgba:
            return color

        valid_mask = (color[:, :, -1])[:, :, np.newaxis]
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]

        output_img = output_img.astype(np.float32)
        return output_img

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9), 
                            rot_axis=[1,0,0], rot_angle=0, is_right=1):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(), vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        
        rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t = None,
            rot=None,
            rot_axis=[1,0,0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
        ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        focal_length = focal_length if focal_length is not None else self.focal_length

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length/render_res[1]])

        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle, is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def render_rgba_multiple(
            self,
            vertices: List[np.array],
            cam_t: List[np.array],
            rot_axis=[1,0,0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
        ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        mesh_list = [pyrender.Mesh.from_trimesh(self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle, is_right=sss)) for vvv,ttt,sss in zip(vertices, cam_t, is_right)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i,mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def render_visible_faces(
        self,
        vertices: List[np.array],
        cam_t: List[np.array],
        visible_faces: List[np.array],  # 新增可见面片参数
        mesh_base_color=(0.7, 0.7, 0.7, 1.0),  # 默认颜色（RGBA）
        highlight_color=(1.0, 0.0, 0.0, 1.0),  # 高亮颜色（RGBA）
        rot_axis=[1,0,0],
        rot_angle=0,
        scene_bg_color=(0,0,0),
        render_res=[256, 256],
        focal_length=None,
        is_right=None,
    ):
        """
        可视化指定可见面片的增强渲染函数
        新增参数：
            visible_faces: 各mesh对应的可见面片索引列表
            highlight_color: 高亮颜色（RGBA格式）
        """
        # 创建离屏渲染器
        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_res[0],
            viewport_height=render_res[1],
            point_size=1.0
        )

        # 处理默认参数
        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        # 创建带可见面片颜色的mesh列表
        mesh_list = []
        for idx, (vvv, ttt, sss) in enumerate(zip(vertices, cam_t, is_right)):
            # 获取基础mesh
            base_mesh = self.vertices_to_trimesh(
                vertices=vvv,
                camera_translation=ttt,
                mesh_base_color=mesh_base_color[:3],  # 转换颜色格式
                rot_axis=rot_axis,
                rot_angle=rot_angle,
                is_right=sss
            )
            
            # 创建面片颜色数组
            face_colors = np.zeros((len(base_mesh.faces), 4))
            face_colors[:] = mesh_base_color  # 设置默认颜色
            face_colors[visible_faces[idx]] = highlight_color  # 设置可见面颜色
            
            # 应用面片颜色
            base_mesh.visual.face_colors = face_colors
            
            # 转换为pyrender mesh（启用面片颜色）
            pyrender_mesh = pyrender.Mesh.from_trimesh(
                base_mesh,
                material=pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    roughnessFactor=0.4,
                    alphaMode='OPAQUE'
                ),
                smooth=False
            )
            mesh_list.append(pyrender_mesh)

        # 创建场景
        scene = pyrender.Scene(
            bg_color=[*scene_bg_color, 0.0],
            ambient_light=[0.5, 0.5, 0.5]  # 增强环境光
        )
        
        # 添加所有mesh到场景
        for i, mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        # 设置相机
        camera_pose = np.eye(4)
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12
        )
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)

        # 优化光照配置
        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        # 执行渲染
        color, rend_depth = renderer.render(
            scene,
            flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT  # 平直着色
        )
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color, rend_depth

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def vis_keypoints(self, img, kps, alpha=1, radius=3, color=None):
        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap("rainbow")
        if color is None:
            colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
            colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # Perform the drawing on a copy of the image, to allow for blending.
        kp_mask = np.copy(img)
        kp_mask = kp_mask.astype(np.uint8)
        # import pdb
        # pdb.set_trace()
        # Draw the keypoints.
        for i in range(len(kps)):
            p = kps[i][0].astype(int), kps[i][1].astype(int)
            if color is None:
                cv2.circle(
                    kp_mask,
                    p,
                    radius=radius,
                    color=colors[i],
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.circle(
                    kp_mask,
                    p,
                    radius=radius,
                    color=color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

        # Blend the keypoints.
        return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

    def render_mesh(self, img, mesh, cam_t, cam_param):
        # to run on cluster where headless pyrender is not supported for A100/V100
        # cam_param = {
        #     "focal": [self.focal_length, self.focal_length],
        #     "princpt": [img.shape[1] / 2., img.shape[0] / 2.]
        # }
        # img = np.transpose(img, (1, 2, 0))
        vertices_2d = perspective_projection(mesh, cam_t, cam_param[0], cam_param[1]).reshape(-1, 2).detach().cpu().numpy()
        # vertices_2d = self.convert_crop_coords_to_orig_img(bbox, vertices_2d, torch.tensor(self.img_res).to(mesh.device))
        img = self.vis_keypoints(img, vertices_2d, alpha=0.8, radius=2, color=(0, 0, 255))
        return img, vertices_2d