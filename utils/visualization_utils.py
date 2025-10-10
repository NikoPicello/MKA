import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh

import torch 
# from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    BlendParams,
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardGouraudShader,
    SoftPhongShader,
    TexturesVertex,
)

def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1, radius=3, color=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    if color is None:
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        if color is None:
            cv2.circle(kp_mask, p, radius=radius, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(kp_mask, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '\n') 
    obj_file.close()

def perspective_projection(vertices, cam_param):
    # vertices: [N, 3]
    # cam_param: [3]
    fx, fy= cam_param['focal']
    cx, cy = cam_param['princpt']
    vertices[:, 0] = vertices[:, 0] * fx / vertices[:, 2] + cx
    vertices[:, 1] = vertices[:, 1] * fy / vertices[:, 2] + cy
    return vertices

def render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False):
    if mesh_as_vertices:
        # to run on cluster where headless pyrender is not supported for A100/V100
        vertices_2d = perspective_projection(vertices, cam_param)
        img = vis_keypoints(img, vertices_2d, alpha=0.8, radius=2, color=(0, 0, 255))
    else:
        focal, princpt = cam_param['focal'], cam_param['princpt']
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        # the inverse is same
        pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        

        # render material
        base_color = (1.0, 193/255, 193/255, 1.0)
        material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0,
                alphaMode='OPAQUE',
                baseColorFactor=base_color)
        
        material_new = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.1,
                roughnessFactor=0.4,
                alphaMode='OPAQUE',
                emissiveFactor=(0.2, 0.2, 0.2),
                baseColorFactor=(0.7, 0.7, 0.7, 1))  
        material = material_new
        
        # get body mesh
        body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
        body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

        # prepare camera and light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        cam_pose = pyrender2opencv @ np.eye(4)
        
        # build scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                        ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        scene.add(body_mesh, 'mesh')

        # render scene
        r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                        viewport_height=img.shape[0],
                                        point_size=1.0)
        
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        alpha = 0.8 # set transparency in [0.0, 1.0]

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        valid_mask = valid_mask * alpha
        img = img / 255
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)
        # output_img = color[:, :, :]

        img = (output_img * 255).astype(np.uint8)
    return img


def render_mesh_pt3d(img, verts, faces, cam_param, rasterizer=None):
    device = verts.device
    img_h, img_w = img.shape[:2]
    image_size = torch.tensor([img_h, img_w]).unsqueeze(0).to(device)
    image_size_wh = image_size.flip(dims=(1, ))
    scale = image_size_wh.min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    focal, princpt = cam_param['focal'], cam_param['princpt']

    focal_length = torch.tensor([focal[0], focal[1]]).float().unsqueeze(0).to(device)
    principal_point = torch.tensor([princpt[0], princpt[1]]).float().unsqueeze(0).to(device)
    focal_pt = focal_length / scale
    p0_pt = -(principal_point - c0) / scale

    camera_pose = torch.eye(4).unsqueeze(0).to(device)
    R_pt = camera_pose[:, :3, :3].clone().permute(0, 2, 1)
    R_pt[:, :, :2] *= -1
    tvec_pt = camera_pose[:, :3, 3].clone()
    tvec_pt[:, :2] *= -1

    cameras = PerspectiveCameras(R=R_pt, T=tvec_pt, focal_length=focal_pt, principal_point=p0_pt, image_size=image_size, device=device)

    blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
    if rasterizer is None: 
        raster_settings = RasterizationSettings(
            image_size=(img_h, img_w), 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size = 0,  # this setting controls whether naive or coarse-to-fine rasterization is used
            max_faces_per_bin = None  # this setting is for coarse rasterization
        )
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
    
    lights = PointLights(device=device, location=((0.0, 2.0, -2.0),), specular_color=((0.0, 0.0, 0.0),))
    gray_renderer = MeshRenderer(
        rasterizer=rasterizer,        
        shader = HardGouraudShader(device=device, lights=lights, cameras=cameras, blend_params=blend_params)
    )

    verts_rgb = torch.ones_like(verts)
    tex = TexturesVertex(verts_features=verts_rgb).to(device)
    meshes = Meshes(verts=verts, faces=faces, textures=tex).to(device)
    rendered_imgs = gray_renderer(meshes_world=meshes, cameras=cameras)

    # rendered_imgs = rendered_imgs[:, :img_h, :img_w].clone() ### crop 
    render_img = rendered_imgs[0, ..., :3].detach().cpu().numpy() * 255.0
    render_mask = rendered_imgs[0, ..., 3:].detach().cpu().numpy() 
    output_image = img * (1.0 - render_mask) + render_img * render_mask
    output_image = output_image.astype(np.uint8)
    return output_image

def get_rasterizer(img_h, img_w):
    cameras = look_at_view_transform(2.7, 10, 20)
    blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        bin_size = 0,  # this setting controls whether naive or coarse-to-fine rasterization is used
        max_faces_per_bin = None  # this setting is for coarse rasterization
    )
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    return rasterizer

def check_visibility_pt3d(rasterizer, img, verts, faces, cam_param):
    device = verts.device
    img_h, img_w = img.shape[:2]
    image_size = torch.tensor([img_h, img_w]).unsqueeze(0).to(device)
    image_size_wh = image_size.flip(dims=(1, ))
    scale = image_size_wh.min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    focal, princpt = cam_param['focal'], cam_param['princpt']
    focal_length = torch.tensor([focal[0], focal[1]]).float().unsqueeze(0).to(device)
    principal_point = torch.tensor([princpt[0], princpt[1]]).float().unsqueeze(0).to(device)
    focal_pt = focal_length / scale
    p0_pt = -(principal_point - c0) / scale

    camera_pose = torch.eye(4).unsqueeze(0).to(device)
    R_pt = camera_pose[:, :3, :3].clone().permute(0, 2, 1)
    R_pt[:, :, :2] *= -1
    tvec_pt = camera_pose[:, :3, 3].clone()
    tvec_pt[:, :2] *= -1
    cameras = PerspectiveCameras(R=R_pt, T=tvec_pt, focal_length=focal_pt, principal_point=p0_pt, image_size=image_size, device=device)
        
    mesh = Meshes(verts=verts, faces=faces).to(device)
    # with torch.no_grad():
    #     fragments = rasterizer(mesh, cameras=cameras)
    #     print("=== fragments", fragments.zbuf.size(), flush=True)
    #     depth_map = fragments.zbuf[0, ..., 0].cpu().numpy() 
    # cv2.imwrite("./check.jpg", depth_map*255)
    # exit(0)
    
    vertices = mesh.verts_packed().cpu().numpy()
    homogeneous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    cam_transform = cameras.get_world_to_view_transform()
    projected_vertices = cam_transform.transform_points(mesh.verts_packed())
    projected_vertices = projected_vertices.cpu().float().numpy()
    screen_vertices = cameras.transform_points_screen(
        mesh.verts_packed(),
        image_size=(img_h, img_w)
    )[:, :2].cpu().numpy().astype(int)

    visibility = np.zeros(len(vertices), dtype=bool)
    min_depth_arr = np.ones((img_h, img_w), dtype=np.int32) * -1
    for i, (x, y) in enumerate(screen_vertices):
        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            visibility[i] = False  
            continue
        # compare vertex depth with depth map (need to consider nan)
        depth = projected_vertices[i, 2]
        if depth < 0: #  behind the camera
            continue 
        if min_depth_arr[y][x] < 0:
            min_depth_arr[y][x] = i 
        else:
            cur_midx = min_depth_arr[y][x]
            cur_mdepth = projected_vertices[cur_midx, 2]
            if depth < cur_mdepth:
                min_depth_arr[y][x] = i

        # if depth < 0:  #  behind the camera 
        #     visibility[i] = False
        #     continue
        # map_depth = depth_map[y, x]
        # if i == 6256 or i == 6634:
        #     print(i, depth, map_depth, flush=True)

        # if np.isnan(map_depth) or depth < map_depth or abs(depth - map_depth) < 1e-6:  # floating error
        #     visibility[i] = True
        # else:
        #     visibility[i] = False

    for i, (x, y) in enumerate(screen_vertices):
        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            continue
        if min_depth_arr[y][x] < 0:
            continue             
        cur_idx = min_depth_arr[y][x]
        visibility[cur_idx] = True 
    
    # for i, v in enumerate(screen_vertices):
    #     # if not visibility[i]:
    #     #     continue
    #     x = int(v[0])
    #     y = int(v[1])
    #     if visibility[i]:
    #         cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    #     # else:
    #     #     cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # cv2.imwrite("./check.jpg", img)
    # print("=== visibility", np.count_nonzero(visibility), len(visibility[visibility == True]) , flush=True)
    # print(np.amax(screen_vertices), np.amin(screen_vertices), flush=True)
    # exit(0)
    return visibility
