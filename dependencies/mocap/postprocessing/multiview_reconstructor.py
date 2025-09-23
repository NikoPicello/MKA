import cv2
import numpy as np
import open3d as o3d


class KinectCamera:

    def __init__(self):
        self.d_height = 0
        self.d_width = 0
        self.c_height = 0
        self.c_width = 0
        self.Kd = np.float32(np.identity(3, dtype=float))
        self.Kc = np.float32(np.identity(3, dtype=float))
        self.Tcalib = np.float32(np.identity(
            4, dtype=float))  # depth_0 -> depth_i
        self.Tglobal = np.float32(np.identity(4,
                                              dtype=float))  # depth_i -> world
        self.Td2c = np.float32(np.identity(4,
                                           dtype=float))  # depth_i -> color_i
        self.center_depth = 0.0

    def __str__(self):
        camera_info = ''
        camera_info += 'd_height: {}, d_width: {}'.format(
            self.d_height, self.d_width)  # noqa: E501
        camera_info += 'c_height: {}, c_width: {}'.format(
            self.c_height, self.c_width)  # noqa: E501
        camera_info += 'Kd: {}'.format(self.Kd)
        camera_info += 'Kc: {}'.format(self.Kc)
        camera_info += 'Tcalib:\n{}'.format(self.Tcalib)
        camera_info += 'Tglobal:\n{}'.format(self.Tglobal)
        camera_info += 'center_depth:\n{}'.format(self.center_depth)
        return camera_info


class Calibrations:

    def __init__(self, smc):
        self.smc = smc
        self.num_device = self.smc.get_num_kinect()
        self.kinects = []
        # init calibrations
        Td02w = self.smc.get_kinect_depth_extrinsics(0)

        for i in range(self.num_device):
            Tdi2w = self.smc.get_kinect_depth_extrinsics(i)

            Td02di = np.linalg.inv(Tdi2w) @ Td02w

            Tci2w = self.smc.get_kinect_color_extrinsics(i)

            Td2c = np.linalg.inv(Tci2w) @ Tdi2w

            c = KinectCamera()
            c.Tcalib = np.float32(Td02di)
            c.Td2c = Td2c
            c.Kd = self.smc.get_kinect_depth_intrinsics(i)[0]
            c.Kc = self.smc.get_kinect_color_intrinsics(i)[0]
            c.d_height, c.d_width = self.smc.get_kinect_depth_resolution(i)
            c.c_height, c.c_width = self.smc.get_kinect_color_resolution(i)

            self.kinects.append(c)

    def update_center_depth(self, camera_id, center_depth):
        self.calibs[camera_id].center_depth = center_depth
        if (camera_id == 0):
            self.update_calibs()

    def update_calibs(self):
        self.calibs[0].Tglobal = np.float32(np.identity(4, dtype=float))
        self.calibs[0].Tglobal[2, 3] = self.calibs[0].center_depth

        for i in range(len(self.calibs) - 1):
            camid = i + 1
            self.calibs[camid].Tglobal = self.calibs[camid].Tcalib
            self.calibs[camid].Tglobal = np.matmul(self.calibs[camid].Tglobal,
                                                   self.calibs[0].Tglobal)

    def scale_calibs(self, scale):
        for i in range(len(self.calibs)):
            self.calibs[i].Tcalib[:3, 3] = self.calibs[i].Tcalib[:3, 3] * scale
            self.calibs[i].Tglobal[:3,
                                   3] = self.calibs[i].Tglobal[:3, 3] * scale
            self.calibs[i].Td2c[:3, 3] = self.calibs[i].Td2c[:3, 3] * scale

    def cut_calibs(self, tar_d_height, tar_d_width):
        for i in range(len(self.calibs)):
            self.calibs[i].Kd[0,
                              2] -= (self.calibs[i].d_width - tar_d_width) / 2
            self.calibs[i].Kd[1, 2] -= (self.calibs[i].d_height -
                                        tar_d_height) / 2


class SingleViewPointCloudReconstructor:

    def __init__(self, smc):
        self.smc = smc
        self.calibrations = Calibrations(smc)
        self.__DILATE_KERNEL_SIZE__ = 5
        self.__EDGE_DILATE_KERNEL_SIZE__ = 5
        self.__CANNY_THRESHOLD__ = 40
        self.__DEPTH_MIN__ = 200
        self.__DEPTH_MAX__ = 3000
        self.__BACKGROUNG_VAL__ = 30000

    def reconstruct(self, device_id, frame_id):

        depth_resolution = self.smc.get_kinect_depth_resolution(device_id)

        cam = o3d.camera.PinholeCameraParameters()
        depth_intrinsics = self.smc.get_kinect_depth_intrinsics(device_id)
        cam.intrinsic.set_intrinsics(depth_resolution[0], depth_resolution[1],
                                     depth_intrinsics[0,
                                                      0], depth_intrinsics[1,
                                                                           1],
                                     depth_intrinsics[0,
                                                      2], depth_intrinsics[1,
                                                                           2])
        cam.extrinsic = np.linalg.inv(
            self.calibrations.kinects[device_id].Tcalib)
        rgb, depth_image = self.smc.get_kinect_rgbd(
            device_id, frame_id, mode='color2depth')

        depth_image[depth_image < self.__DEPTH_MIN__] = self.__BACKGROUNG_VAL__
        depth_image[depth_image > self.__DEPTH_MAX__] = self.__BACKGROUNG_VAL__
        depth_image[self.smc.get_depth_mask(device_id, frame_id) ==
                    0] = self.__BACKGROUNG_VAL__
        edge_mask = self.__edge_filter__(depth_image)
        dialate_mask = self.__dilate_filter__(depth_image)
        mask = np.bitwise_or(edge_mask, dialate_mask)
        mask = np.bitwise_or(
            mask,
            self.smc.get_depth_mask(device_id, frame_id) < 100)
        depth_image = np.where(
            mask,
            np.ones(depth_image.shape, dtype=np.uint16) * 30000, depth_image)
        depth_image = o3d.geometry.Image(depth_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth_image),
            depth_trunc=5.0,
            convert_rgb_to_intensity=False)
        pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, cam.intrinsic)
        pointcloud, ind = pointcloud.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=1.0)
        pointcloud, ind = pointcloud.remove_radius_outlier(
            nb_points=5, radius=0.02)
        pointcloud.estimate_normals(fast_normal_computation=False)
        pointcloud.orient_normals_towards_camera_location()
        return pointcloud

    def __dilate_filter__(self, image):
        kernel = np.ones(
            (self.__DILATE_KERNEL_SIZE__, self.__DILATE_KERNEL_SIZE__),
            np.uint8)
        depth_mask = (image == 30000).astype(np.uint8) * 255
        depth_mask = cv2.dilate(depth_mask, kernel, iterations=1)
        depth_mask = depth_mask > 128
        return depth_mask

    def __depth_canny__(self, image):
        # histogram equalization
        hist, _ = np.histogram(image.flatten(), 30001, [0, 30001])
        cdf = hist.cumsum()
        cdf = cdf * hist.max() / cdf.max()
        image = cdf[image]
        image = image / 30000.0 * 255.0
        edged = cv2.Canny(
            image.astype(np.uint8), self.__CANNY_THRESHOLD__,
            self.__CANNY_THRESHOLD__)
        return edged

    def __edge_filter__(self, image):
        edge = self.__depth_canny__(image)
        dilate_kernel = np.ones((self.__EDGE_DILATE_KERNEL_SIZE__,
                                 self.__EDGE_DILATE_KERNEL_SIZE__), np.uint8)
        depth_image_mask = cv2.dilate(edge, dilate_kernel, iterations=1)
        return (depth_image_mask > 128)


class MultiViewReconstructor:

    def __init__(self, smc):
        self.smc = smc
        self.single_view_reconstructor = SingleViewPointCloudReconstructor(smc)

    def reconstruct_pointcloud(self, frame_id):
        pointcloud = o3d.geometry.PointCloud()

        for device_id in range(self.smc.get_num_kinect()):
            extrinsic = np.linalg.inv(
                np.linalg.inv(self.smc.get_kinect_depth_extrinsics(device_id))
                @ self.smc.get_kinect_depth_extrinsics(0))
            pointcloud__ = self.single_view_reconstructor.reconstruct(
                device_id, frame_id).transform(extrinsic)
            pointcloud += pointcloud__
        return pointcloud

    def reconstruct_mesh_poisson(self, frame_id, depth=9):
        pointcloud = self.reconstruct_pointcloud(frame_id)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pointcloud, depth=depth)
        return mesh
