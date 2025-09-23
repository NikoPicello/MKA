import threading

from .utils import process_mmdet_results


class MMDetThread(threading.Thread):

    def __init__(self, threadID, name, model, inference_detector_func,
                 input_condition, output_condition, next_thread):
        """A thread detects images in batch.

        Args:
            threadID (int):
                ID of the thread.
            name (str):
                Name of the thread.
            model (nn.Module):
                An mmdet detector constructed by mmdet.apis.init_detector
            inference_detector_func (function):
                inference_detector(model, imgs) defined in mmdet.apis
            input_condition (threading.Condition):
                Condition lock between this thread and the thread before it.
            output_condition (threading.Condition):
                Condition lock between this thread and the thread after it.
            next_thread (threading.Thread):
                The thread after this thread. An MMPoseThread is expected.
        """
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.model = model
        self.input = None
        self.output = None
        self.inference_detector_func = inference_detector_func
        self.input_condition = input_condition
        self.output_condition = output_condition
        self.next_thread = next_thread
        self.stop_flag = False

    def run(self):
        """If self.input in not None, run mmdet once.

        Args:
            Input in self.input:
                img_np_batch (list):
                    A list of images. Each item is read by cv2.imread.
                frame_name_batch (list):
                    A list of names. Each item is an str.
                last_batch (bool):
                    If this input is the last batch, set to True.
                    The thread will set self.to_output, clear self.output and
                    notify the follwing process.
                    Otherwise, set to False.
        Returns:
            Output in self.output:
                img_np_batch (list):
                    Equals to self.input["img_np_batch"].
                frame_name_batch (list):
                    Equals to self.input["frame_name_batch"].
                person_results (list):
                    A list of detected bounding boxes.
                    Each item is a dict with no more than one bbox.
                last_batch (bool):
                    Equals to self.input["last_batch"].
        """
        while True:
            if self.stop_flag:
                print('mmdet thread stops normally')
                break
            if self.input_condition.acquire():
                if self.input is None:
                    self.input_condition.wait(timeout=3)
                else:
                    try:
                        # test a batch of images,
                        # the resulting box is (x1, y1, x2, y2)
                        mmdet_results = \
                            self.inference_detector_func(
                                self.model, self.input['img_np_batch'])
                        # len(mmdet_results) # a list of batch_size
                        # len(mmdet_results[0]) # a list of size 80

                        # keep the person class bounding boxes.
                        person_results = \
                            process_mmdet_results(mmdet_results)
                        self.output = \
                            [
                                self.input['img_np_batch'],
                                self.input['frame_name_batch'],
                                person_results,
                                self.input['last_batch']
                            ]
                        self.input = None
                        if self.output_condition.acquire():
                            self.next_thread.input = {
                                'img_np_batch': self.output[0],
                                'frame_name_batch': self.output[1],
                                'person_results': self.output[2],
                                'last_batch': self.output[3],
                            }
                            self.output_condition.notify()
                            self.output_condition.release()
                    except Exception as e:
                        print('mmdet thread catch an exception and exit:')
                        print(e)
                        print('File: ',
                              e.__traceback__.tb_frame.f_globals['__file__'])
                        print('Line: ', e.__traceback__.tb_lineno)
                        self.input = None
                        self.stop_flag = True
                        self.input_condition.release()
                        break
                self.input_condition.release()


class MMPoseThread(threading.Thread):

    def __init__(self, threadID, name, model, bbox_thresh, format, dataset,
                 inference_batch_func, return_heatmap, outputs,
                 input_condition, output_condition):
        """A thread detects poses in batch.

        Args:
            threadID (int):
                ID of the thread.
            name (str):
                Name of the thread.
            model (nn.Module):
                An mmdet detector constructed by mmpose.apis.init_pose_model
            bbox_thresh (float):
                Threshold of a bbox. Those have lower scores will be ignored.
                0.0 is recommended for mocap.
            format (str):
                Format of input bbox, xyxy or xywh.
                xyxy is recommended for mmdet results.
            dataset (str):
                Typically it is set to model.cfg.data['test']['type'].
            inference_batch_func (function):
                inference_top_down_pose_model_batch in mmpose.apis
            return_heatmap (bool):
                Whther to return heatmap.
                False is recommended for mocap.
            outputs (list or None):
                List of layer names after which
                the feature map will be returned.
                None is recommended for mocap.
            input_condition (threading.Condition):
                Condition lock between this thread and the thread before it.
            output_condition (threading.Condition):
                Condition lock between this thread and the thread after it.
        """
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.model = model

        self.infer_bbox_thresh = bbox_thresh
        self.infer_format = format
        self.infer_dataset = dataset
        self.infer_return_heatmap = return_heatmap
        self.infer_outputs = outputs
        self.inference_batch_func = inference_batch_func

        self.input = None
        self.output = None
        self.to_output = None
        self.input_condition = input_condition
        self.output_condition = output_condition
        self.stop_flag = False
        self.last_batch = False

    def run(self):
        """If self.input in not None, run mmpose once.

        Args:
            Input in self.input:
                img_np_batch (list):
                    A list of images. Each item is read by cv2.imread.
                frame_name_batch (list):
                    A list of names. Each item is an str.
                person_results (list):
                    A list of detected bounding boxes.
                    Each item is a dict with no more than one bbox.
                last_batch (bool):
                    If this input is the last batch, set to True.
                    The thread will set self.to_output, clear self.output and
                    notify the follwing process.
                    Otherwise, set to False.
        Returns:
            Output in self.output (dict):
                key (frame name in frame_name_batch):
                    value (pose result)
                Details can be found in MMPoseThread.infer_pose().
        """
        while True:
            if self.stop_flag:
                print('mmpose thread stops normally')
                break
            if self.last_batch:
                if self.output_condition.acquire():
                    self.output_condition.notify()
                    self.output_condition.release()
            if self.input_condition.acquire():
                if self.input is None:
                    self.input_condition.wait(timeout=3)
                else:
                    try:
                        self.infer_pose()
                        last_batch = self.input['last_batch']
                        self.input = None
                        if last_batch:
                            if self.output_condition.acquire():
                                self.to_output = self.output
                                self.last_batch = True
                                self.output = None
                                self.output_condition.notify()
                                self.output_condition.release()
                    except Exception as e:
                        print('mmpose thread catch an exception and exit:')
                        print(e)
                        self.input = None
                        self.stop_flag = True
                        self.input_condition.release()
                        break
                self.input_condition.release()

    def infer_pose(self):
        """Infer images in self.input, put the result into self.output.

        Each value in self.output is a dict of two keys (bbox, keypoints). bbox
        (xyxy, score) in shape [5,], and keypoints (keypoint index, x+y+score)
        in shape [133, 3]
        """
        pose_results = self.inference_batch_func(
            self.model,
            self.input['img_np_batch'],
            self.input['person_results'],
            bbox_thr=self.infer_bbox_thresh,
            format=self.infer_format,
            dataset=self.infer_dataset,
            return_heatmap=self.infer_return_heatmap,
            outputs=self.infer_outputs)
        if self.output is None:
            self.output = {}
        for batch_index in range(len(self.input['img_np_batch'])):
            batch_item_frame_name = self.input['frame_name_batch'][batch_index]
            self.output[batch_item_frame_name] = pose_results[batch_index]
            # print(f"{batch_item_frame_name}:{len(pose_results[batch_index])}")
