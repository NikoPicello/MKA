import threading

import cv2
import numpy as np
from tqdm import tqdm

from zoehuman.core.conventions.keypoints_mapping import (  # noqa:E501
    KEYPOINTS_FACTORY, convert_kps,
)
from zoehuman.data.data_structures.human_data import HumanData
from zoehuman.utils.ffmpeg_utils import video_to_array
from .detection_thread import MMDetThread, MMPoseThread  # noqa:E501
from .utils import process_mmdet_results

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
assert has_mmdet, 'Please install mmdet to run detection.'
try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
    has_mmpose = True
except (ImportError, ModuleNotFoundError):
    has_mmpose = False
assert has_mmpose, 'Please install mmpose to run detection.'
try:
    from mmpose.apis import inference_top_down_pose_model_batch
    has_mmpose_batch = True
except (ImportError, ModuleNotFoundError):
    has_mmpose_batch = False


def get_pose_detector(detector_kwargs,
                      pose_model_kwargs,
                      bbox_thr=0.0,
                      max_batch_size=48,
                      verbose=True):
    """Get a pose detector according to environment.

    Args:
        detector_kwargs (dict):
            A dict contains args of mmdet.apis.init_detector.
            Necessary keys: config, checkpoint
        pose_model_kwargs (dict):
            A dict contains args of mmpose.apis.init_pose_model.
            Necessary keys: config, checkpoint
        bbox_thr (float, optional):
            Threshold of a bbox. Those have lower scores will be ignored.
            0.0 is recommended for mocap.
            Defaults to 0.0.
        max_batch_size (int, optional):
            Max size of a batch. 48 is recommeded for one V100.
            Defaults to 48.
            Only valid when PoseDetectorBatch is supported.
        verbose (bool, optional):
            Whether to print the type of returned detector.
            Defaults to True.

    Returns:
        PoseDetectorBatch or PoseDetector
    """
    ret_instance = None
    if has_mmpose_batch:
        ret_instance = \
            PoseDetectorBatch(
                detector_kwargs, pose_model_kwargs,
                bbox_thr=bbox_thr, max_batch_size=max_batch_size
            )
    else:
        ret_instance = \
            PoseDetector(
                detector_kwargs, pose_model_kwargs,
                bbox_thr=bbox_thr
            )
    if verbose:
        print(f'Type of PoseDetector: {type(ret_instance)}')
    return ret_instance


class PoseDetectorBatch:

    def __init__(self,
                 detector_kwargs,
                 pose_model_kwargs,
                 bbox_thr=0.0,
                 max_batch_size=48) -> None:
        """Init a detector and a pose model.

        Args:
            detector_kwargs (dict):
                A dict contains args of mmdet.apis.init_detector.
                Necessary keys: config, checkpoint
            pose_model_kwargs (dict):
                A dict contains args of mmpose.apis.init_pose_model.
                Necessary keys: config, checkpoint
            bbox_thr (float, optional):
                Threshold of a bbox. Those have lower scores will be ignored.
                0.0 is recommended for mocap.
                Defaults to 0.0.
            max_batch_size (int, optional):
                Max size of a batch.
                48 is recommeded for one V100.
                Defaults to 48.
        """
        # build the detector from a config file and a checkpoint file
        self.det_model = init_detector(**detector_kwargs)
        # build the pose model from a config file and a checkpoint file
        self.pose_model = init_pose_model(**pose_model_kwargs)
        self.main_mmdet_lock = threading.Condition()
        self.mmdet_mmpose_lock = threading.Condition()
        self.mmpose_main_lock = threading.Condition()
        self.mmpose_thread = MMPoseThread(
            threadID=1,
            name='mmpose1',
            model=self.pose_model,
            bbox_thresh=bbox_thr,
            format='xyxy',
            dataset=self.pose_model.cfg.data['test']['type'],
            inference_batch_func=inference_top_down_pose_model_batch,
            return_heatmap=False,
            outputs=None,
            input_condition=self.mmdet_mmpose_lock,
            output_condition=self.mmpose_main_lock)
        self.mmdet_thread = MMDetThread(
            threadID=0,
            name='mmdet0',
            model=self.det_model,
            inference_detector_func=inference_detector,
            input_condition=self.main_mmdet_lock,
            output_condition=self.mmdet_mmpose_lock,
            next_thread=self.mmpose_thread)
        self.bbox_thr = bbox_thr
        self.max_batch_size = max_batch_size
        self.mmpose_thread.start()
        self.mmdet_thread.start()

    def get_data_source_name(self):
        """Get data_source from dataset type in config file of the pose model.

        Returns:
            str:
                Name of the data_source. Must be
                a key of KEYPOINTS_FACTORY.
        """
        return __translate_data_source__(
            self.pose_model.cfg.data['test']['type'])

    def __del__(self):
        if self.main_mmdet_lock.acquire():
            self.mmdet_thread.stop_flag = True
            print('Main thread notifies mmdet to stop')
            self.main_mmdet_lock.notify()
            self.main_mmdet_lock.release()
        if self.mmdet_mmpose_lock.acquire():
            self.mmpose_thread.stop_flag = True
            print('Main thread notifies mmpose to stop')
            self.mmdet_mmpose_lock.notify()
            self.mmdet_mmpose_lock.release()
        del self.det_model
        del self.pose_model

    def __push_batch__(self, img_np_batch, frame_name_batch, last_batch):
        if self.mmdet_thread.stop_flag or self.mmpose_thread.stop_flag:
            print('Det thread stops by exception, return.')
            return
        if self.main_mmdet_lock.acquire():
            self.mmdet_thread.input = {
                'img_np_batch': img_np_batch,
                'frame_name_batch': frame_name_batch,
                'last_batch': last_batch,
            }
            self.main_mmdet_lock.notify()
            self.main_mmdet_lock.release()

    def __fetch_result__(self):
        if self.mmpose_main_lock.acquire():
            if not self.mmpose_thread.last_batch:
                self.mmpose_main_lock.wait(timeout=3)
                self.mmpose_main_lock.release()
                return None
            else:
                ret_dict = self.mmpose_thread.to_output
                self.mmpose_thread.last_batch = False
                self.mmpose_thread.to_output = None
                self.mmpose_main_lock.release()
                return ret_dict
        return None

    def infer_array(self, frame_ndarray_dict, disable_tqdm=False):
        """Infer frames already in memory(ndarray type).

        Args:
            frame_ndarray_dict (dict):
                A dict whose keys are frame names and
                values are frames' ndarray read by cv2.imread.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.

        Returns:
            dict:
                Keys equal to frame_path_dict.keys(),
                values are pose results in dict type.
                Each value is a dict of two keys (bbox, keypoints).
                bbox (xyxy, score) in shape [5,],
                and keypoints (keypoint index, x+y+score) in shape [133, 3]
        """
        image_batch_list = []
        name_batch_list = []
        frame_count = 0
        for frame_name, frame_ndarray in tqdm(
                frame_ndarray_dict.items(), disable=disable_tqdm):
            image_batch_list.append(frame_ndarray)
            name_batch_list.append(frame_name)
            frame_count += 1
            if len(image_batch_list) >= self.max_batch_size:
                last_batch = (frame_count == len(frame_ndarray_dict))
                self.__push_batch__(image_batch_list, name_batch_list,
                                    last_batch)
                # clean for next batch
                image_batch_list = []
                name_batch_list = []
        # deal with the rest
        if len(image_batch_list) > 0:
            last_batch = True
            self.__push_batch__(image_batch_list, name_batch_list, last_batch)
        ret_dict = None
        while ret_dict is None:
            ret_dict = self.__fetch_result__()
        return ret_dict

    def infer_frames(self, frame_path_dict, disable_tqdm=False):
        """Infer frames from file.

        Args:
            frame_path_dict (dict):
                A dict whose keys are frame names and
                values are frames' absolute paths.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.

        Returns:
            dict:
                Keys equal to frame_path_dict.keys(),
                values are pose results in dict type.
                Details can be found in PoseDetectorBatch.infer_array().
        """
        ndarray_dict = {}
        for frame_name, frame_abs_path in frame_path_dict.items():
            img_np = cv2.imread(frame_abs_path)
            ndarray_dict[frame_name] = img_np
        ret_dict = self.infer_array(ndarray_dict, disable_tqdm)
        return ret_dict

    def infer_video(self, video_path, disable_tqdm=False):
        """Infer frames from a video file.

        Args:
            video_path (str):
                Path to the video to be detected.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.

        Returns:
            dict:
                Keys equal to frame_path_dict.keys(),
                values are pose results in dict type.
                Details can be found in PoseDetectorBatch.infer_array().
        """
        frames_array = video_to_array(video_path)
        ndarray_dict = {}
        for frame_index in range(frames_array.shape[0]):
            frame_name = 'frame_%06d.jpg' % frame_index
            ndarray_dict[frame_name] = frames_array[frame_index]
        ret_dict = self.infer_array(ndarray_dict, disable_tqdm)
        return ret_dict


class PoseDetector:

    def __init__(self,
                 detector_kwargs,
                 pose_model_kwargs,
                 bbox_thr=0.0) -> None:
        """Init a detector and a pose model.

        Args:
            detector_kwargs (dict):
                A dict contains args of mmdet.apis.init_detector.
                Necessary keys: config, checkpoint
            pose_model_kwargs (dict):
                A dict contains args of mmpose.apis.init_pose_model.
                Necessary keys: config, checkpoint
            bbox_thr (float, optional):
                Threshold of a bbox. Those have lower scores will be ignored.
                0.0 is recommended for mocap. Defaults to 0.0.
        """
        # build the detector from a config file and a checkpoint file
        self.det_model = init_detector(**detector_kwargs)
        # build the pose model from a config file and a checkpoint file
        self.pose_model = init_pose_model(**pose_model_kwargs)
        self.bbox_thr = bbox_thr

    def get_data_source_name(self):
        """Get data_source from dataset type in config file of the pose model.

        Returns:
            str:
                Name of the data_source. Must be
                a key of KEYPOINTS_FACTORY.
        """
        return __translate_data_source__(
            self.pose_model.cfg.data['test']['type'])

    def __del__(self):
        del self.det_model
        del self.pose_model

    def infer_array(self,
                    frame_ndarray_dict,
                    disable_tqdm=False,
                    multi_person=False):
        """Infer frames already in memory(ndarray type).

        Args:
            frame_ndarray_dict (dict):
                A dict whose keys are frame names and
                values are frames' ndarray read by cv2.imread.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            multi_person (bool, optional):
                Whether to allow multi-person detection, which is
                slower than single-person.
                Defaults to False.

        Returns:
            dict:
                Keys equal to frame_path_dict.keys(),
                values are pose results in dict type.
                Each value is a dict of two keys (bbox, keypoints).
                bbox (xyxy, score) in shape [5,],
                and keypoints (keypoint index, x+y+score) in shape [133, 3]
        """
        ret_dict = {}
        for frame_name, frame_ndarray in tqdm(
                frame_ndarray_dict.items(), disable=disable_tqdm):
            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = [
                inference_detector(self.det_model, frame_ndarray),
            ]
            # keep the person class bounding boxes.
            person_results = \
                process_mmdet_results(mmdet_results, multi_person=multi_person)
            # test a single image, with a list of bboxes.
            pose_results, _ = inference_top_down_pose_model(
                self.pose_model,
                frame_ndarray,
                person_results,
                bbox_thr=self.bbox_thr,
                format='xyxy',
                dataset=self.pose_model.cfg.data['test']['type'],
                return_heatmap=False,
                outputs=None)
            ret_dict[frame_name] = pose_results
        return ret_dict

    def infer_frames(self,
                     frame_path_dict,
                     disable_tqdm=False,
                     multi_person=False):
        """Infer frames from file.

        Args:
            frame_path_dict (dict):
                A dict whose keys are frame names and
                values are frames' absolute paths.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            multi_person (bool, optional):
                Whether to allow multi-person detection, which is
                slower than single-person.
                Defaults to False.

        Returns:
            dict:
                Keys equal to frame_path_dict.keys(),
                values are pose results in dict type.
                Details can be found in PoseDetector.infer_array().
        """
        ndarray_dict = {}
        for frame_name, frame_abs_path in frame_path_dict.items():
            if isinstance(frame_abs_path, str):
                img_np = cv2.imread(frame_abs_path)
            else:
                img_np = frame_abs_path
            ndarray_dict[frame_name] = img_np
        ret_dict = self.infer_array(ndarray_dict, disable_tqdm, multi_person)
        return ret_dict

    def infer_video(self, video_path, disable_tqdm=False, multi_person=False):
        """Infer frames from a video file.

        Args:
            video_path (str):
                Path to the video to be detected.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            multi_person (bool, optional):
                Whether to allow multi-person detection, which is
                slower than single-person.
                Defaults to False.

        Returns:
            dict:
                Keys equal to frame_path_dict.keys(),
                values are pose results in dict type.
                Details can be found in PoseDetector.infer_array().
        """
        frames_array = video_to_array(video_path)
        ndarray_dict = {}
        for frame_index in range(frames_array.shape[0]):
            frame_name = 'frame_%06d.jpg' % frame_index
            ndarray_dict[frame_name] = frames_array[frame_index]
        ret_dict = self.infer_array(
            ndarray_dict, disable_tqdm, multi_person=False)
        return ret_dict


def __translate_data_source__(mmpose_dataset_name):
    if mmpose_dataset_name == 'TopDownSenseWholeBodyDataset':
        return 'sense_whole_body'
    elif mmpose_dataset_name == 'TopDownCocoWholeBodyDataset':
        return 'coco_wholebody'
    else:
        raise NotImplementedError


def convert_results_to_human_data(infer_result,
                                  src_human_data=None,
                                  data_source='coco_wholebody',
                                  data_destination='human_data',
                                  bbox_threshold=0.0,
                                  person_index=0):
    """Convert value returned by PoseDetector(Batch).infer_* to HumanData dict.

    Args:
        infer_result (dict):
            Return value of functions PoseDetector(Batch).infer_*.
        src_human_data (HumanData, optional):
            When src_human_data is set,
            the HumanData to return will inherit values from src_human_data,
            except bbox_xywh, keypoints2d and mask,
            which are predicted by PoseDetector.
            Defaults to None.
        data_source (str, optional):
            Source data type from keypoints_factory
            Defaults to coco_wholebody.
        data_source (str, optional):
            Destination data type from keypoints_factory
            Defaults to human_data.
        bbox_threshold (float, optional):
            The score threshold to filter bad pose with bbox score.
            If the person at person_index has a bbox score
            under bbox_threshold,
            its keypoints2d in this frame will be set to zeros.
            Defaults to 0.0.
        person_index (int, optional): [description]. Defaults to 0.
    Returns:
        dict:
            HumanData containing bbox_xywh and keypoints2d, as well as values
            inherit from src_human_data.
    """
    ret_human_data_dict = HumanData()
    if src_human_data is not None:
        ret_human_data_dict.update(src_human_data)
    bbox_xywh = np.zeros(shape=[len(infer_result), 5])
    keypoints_number = len(KEYPOINTS_FACTORY[data_destination])
    keypoints2d = np.zeros(shape=[len(infer_result), keypoints_number, 3])
    frame_count = 0
    for _, result_dict_list in infer_result.items():
        if person_index < len(result_dict_list):
            if 'bbox' in result_dict_list[person_index].keys():
                bbox_xyxy_frame = result_dict_list[person_index]['bbox']
                bbox_score = bbox_xyxy_frame[4]
                x = min(bbox_xyxy_frame[0], bbox_xyxy_frame[2])
                y = min(bbox_xyxy_frame[1], bbox_xyxy_frame[3])
                w = abs(bbox_xyxy_frame[0] - bbox_xyxy_frame[2])
                h = abs(bbox_xyxy_frame[1] - bbox_xyxy_frame[3])
            else:
                bbox_score = 1.
                x = y = w = h = 0
            if bbox_score >= bbox_threshold:
                whole_body_keypoints2d = \
                    result_dict_list[person_index]['keypoints'].copy()
            else:
                bbox_score = 0
                whole_body_keypoints2d = np.zeros(
                    shape=result_dict_list[person_index]['keypoints'].shape,
                    dtype=result_dict_list[person_index]['keypoints'].dtype)
        else:
            bbox_score = 0
            x = y = w = h = 0
            whole_body_keypoints2d = np.zeros(
                shape=(len(KEYPOINTS_FACTORY[data_source]), 3), )
        bbox_xywh[frame_count, :] = np.asarray((x, y, w, h, bbox_score))
        whole_body_keypoints2d = np.expand_dims(whole_body_keypoints2d, axis=0)
        human_data_keypoints2d, human_data_mask = convert_kps(
            whole_body_keypoints2d, src=data_source, dst=data_destination)
        keypoints2d[frame_count, :, :] = human_data_keypoints2d[0, :, :]
        frame_count += 1
    ret_human_data_dict['bbox_xywh'] = bbox_xywh
    ret_human_data_dict['keypoints2d'] = keypoints2d
    ret_human_data_dict['keypoints2d_mask'] = human_data_mask
    return ret_human_data_dict
