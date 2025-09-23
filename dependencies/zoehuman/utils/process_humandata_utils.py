import io
import json
from datetime import datetime

import h5py
import numpy as np


class Logger:

    @staticmethod
    def format_time(t: float):
        return f'{t:6.2f}'

    @staticmethod
    def get_current_time():
        return datetime.now().strftime('%d-%b-%Y (%H:%M:%S)')

    @staticmethod
    def log_info(info: dict, path: str):
        """ dump the recorded infomation into json
            and save it
        Args:
            info (dict):
                the infomation dict
            path (str):
                path to the log file
        """
        with open(path, 'w+') as f:
            str_out = Logger.get_json_str_from_dict(info)
            f.write(str_out)
            f.write('\n')

    @staticmethod
    def get_json_str_from_dict(dic: dict):
        str_out = json.dumps(dic, indent=4, sort_keys=True, ensure_ascii=False)
        return str_out


class H5Helper:
    """Some helper function related to h5py can be found here."""
    h5_element_type = \
        (np.int64, np.float64, str, np.float, float, np.float32, int)
    h5_list_type = \
        (list, np.ndarray)

    @staticmethod
    def save_attrs_to_h5file(h5file: h5py.File,
                             root_key: str = '/',
                             dic: dict = {}):
        for k, v in dic.items():
            h5file[root_key].attrs[k] = v
        return None

    @staticmethod
    def recursively_save_dict_contents_to_h5file(h5file: h5py.File,
                                                 root_key: str = '/',
                                                 dic: dict = {}):
        if not root_key.endswith('/'):
            root_key = root_key + '/'
        for k, v in dic.items():
            k = str(k)
            if k == 'attrs':
                if root_key not in h5file:
                    h5file.create_group(root_key)
                H5Helper.save_attrs_to_h5file(h5file, root_key, v)
                continue
            if isinstance(v, dict):
                H5Helper.recursively_save_dict_contents_to_h5file(
                    h5file, root_key + k + '/', v)
            elif isinstance(v, H5Helper.h5_element_type):
                h5file[root_key + k] = v
            elif isinstance(v, H5Helper.h5_list_type):
                try:
                    h5file[root_key + k] = v
                except TypeError:
                    v = np.array(v).astype('|S9')
                    h5file[root_key + k] = v
            else:
                raise TypeError(f'Cannot save {type(v)} type.')
        return None

    @staticmethod
    def load_dict_from_hdf5(filename):
        with h5py.File(filename, 'r') as h5file:
            return H5Helper.recursively_load_dict_contents_from_group(
                h5file, '/')

    @staticmethod
    def load_h5group_attr_to_dict(h5file: h5py.File,
                                  root_key: str = '/') -> dict:
        ans = {}
        for k, v in h5file[root_key].attrs.items():
            ans[k] = v
        return ans

    @staticmethod
    def recursively_load_dict_contents_from_group(h5file: h5py.File,
                                                  root_key: str = '/'):
        if not root_key.endswith('/'):
            root_key = root_key + '/'
        ans = {}
        if len(h5file[root_key].attrs) > 0:
            ans['attrs'] = H5Helper.load_h5group_attr_to_dict(h5file, root_key)
        for key, item in h5file[root_key].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[...]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = \
                    H5Helper.recursively_load_dict_contents_from_group(
                    h5file, root_key + key + '/')
        return ans

    @staticmethod
    def h5py_to_binary(h5f):
        bio = io.BytesIO()
        with h5py.File(bio, 'w') as biof:
            for key, value in h5f.items():
                h5f.copy(
                    value,
                    biof,
                    expand_soft=True,
                    expand_external=True,
                    expand_refs=True)
        return bio


class PipeLine2DHelper:
    """Some helper function can be found here."""

    @staticmethod
    def switch_color_channel(in_frames: np.ndarray,
                             in_tuple: tuple = (0, 2),
                             out_tuple: tuple = (2, 0)):
        """ Switch color channels
            Not an inplace operation
        Args:
            in_frames (np.ndarray):
                the input array of which the
                color channels will be switched
            in_tuple (tuple):
                specify the in channel indices
            out_tuple (tuple):
                specify the out channel indices
        """
        out_frames = in_frames.copy()
        out_frames[..., out_tuple] = in_frames[..., in_tuple]
        return out_frames

    @staticmethod
    def frame_array_to_dict(in_frames: np.ndarray):
        """Store the input frames into a dict."""
        frame_dict = {}
        num_of_frames = in_frames.shape[0]
        for i in range(num_of_frames):
            frame_dict[f'{i:05d}'] = in_frames[i]
        return frame_dict

    @staticmethod
    def pose_dict_to_array(pose_dict: dict):
        """Store the pose dict into a numpy array."""
        ks = sorted(pose_dict.keys())
        pose_array = []
        error_keys = []
        shape = None
        dtype = None
        """
        Travel pose dict to find one frame with meaningful key points
        If the frame exists, record its shape and dtype,
        for those frames with no pose detected, fill the corresponding
        poses with zeros.
        If there is not any frame with meaningful key points, return
        all zeros
        """

        for k in ks:
            try:
                shape = pose_dict[k][0]['keypoints'].shape
                dtype = pose_dict[k][0]['keypoints'].dtype
                break
            except (IndexError, KeyError):
                continue
        if shape is None:
            # todo, replace 133 with a const exp
            return np.zeros((len(k), 133, 3), dtype=np.float32), ks
        for k in ks:
            try:
                pose_array.append(pose_dict[k][0]['keypoints'])
            except (IndexError, KeyError):
                error_keys.append(k)
                pose_array.append(np.zeros(shape=shape, dtype=dtype))
                continue
        return np.array(pose_array), error_keys
