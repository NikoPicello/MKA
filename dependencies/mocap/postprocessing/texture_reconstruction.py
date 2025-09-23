import os
import tempfile

import cv2
import numpy as np
import open3d as o3d
from mocap.postprocessing.multiview_reconstructor import MultiViewReconstructor


class TextureReconstructor:

    def __init__(self, smc):
        self.smc = smc

    def run_texture_projection(self, frame_id, mesh=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            print(temp_dir)
            print('Getting Mesh...')
            if mesh:
                print('Mesh found, exporting...')
            else:
                print('Mesh not found, reconstructing...')
                mv_constructor = MultiViewReconstructor(  # noqa: E501
                    self.smc)
                mesh = mv_constructor.reconstruct_mesh_poisson(frame_id)

            o3d.io.write_triangle_mesh(
                os.path.join(temp_dir, 'mesh.ply'), mesh)

            print('Generating RGB in Depth space...')
            self.__export_color_image__(frame_id, temp_dir)
            print('Generating camera projection parameters...')
            self.__export_texture_parameters__(frame_id, temp_dir)
            os.system(
                'texrecon --keep_unseen_faces --skip_global_seam_leveling --skip_local_seam_leveling {} {}/mesh.ply {}/mesh_textured'  # noqa: E501
                .format(temp_dir, temp_dir, temp_dir))
            textured_mesh = o3d.io.read_triangle_mesh(
                os.path.join(temp_dir, 'mesh_textured.obj'))
            return textured_mesh

    def __export_texture_parameters__(self, frame_id, temp_dir):
        for device_id in range(self.smc.get_num_kinect()):
            depth_resolution = self.smc.get_kinect_depth_resolution(device_id)
            with open(
                    '{}/frame{}_view{}_rgb.cam'.format(temp_dir, frame_id,
                                                       device_id), 'w') as f:
                extrinsics = np.linalg.inv(
                    self.smc.get_kinect_depth_extrinsics(
                        device_id)) @ self.smc.get_kinect_depth_extrinsics(0)
                intrinsecs = self.smc.get_kinect_depth_intrinsics(device_id)
                translation = extrinsics[:3, 3]
                # Write Translation
                f.write('%f %f %f ' %
                        (translation[0], translation[1], translation[2]))
                # Write Rotation
                rotation = extrinsics[:3, :3]
                f.write(' '.join(
                    [str(rotation[i, j]) for i in range(3) for j in range(3)]))
                f.write('\n')
                # Write focal
                focal = intrinsecs[0, 0] + intrinsecs[1, 1]
                f.write('%f ' % (focal / 2.0 / depth_resolution[0]))
                # Write d0 d1 p_aspect
                f.write('0 0 1 ')
                f.write('%f %f' % (intrinsecs[0, -1] / depth_resolution[0],
                                   intrinsecs[1, -1] / depth_resolution[1]))

    def __export_color_image__(self, frame_id, temp_dir):
        for device_id in range(self.smc.get_num_kinect()):
            rgb, _ = self.smc.get_kinect_rgbd(
                device_id, frame_id, mode='color2depth')
            cv2.imwrite(
                '{}/frame{}_view{}_rgb.jpeg'.format(temp_dir, frame_id,
                                                    device_id),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
