def process_mmdet_results(mmdet_results_list, cat_id=0, multi_person=True):
    """Process mmdet results, and return a list of bboxes.

    Args:
        mmdet_results_list (list):
            Result of mmdet.apis.inference_detector
            when the input is a batch. Each item in
            this list is the detection result of an image.
        cat_id (int, optional):
            Category ID. This function will only focus on
            the selected category.
            Defaults to 0.
        multi_person (bool, optional):
            Whether to allow multi-person detection, which is
            slower than single-person.
            Defaults to True.

    Returns:
        list:
            A list of detected bounding boxes. Each item is
            a dict with no more than one bbox.
    """
    ret_list = []
    only_max_arg = not multi_person
    for mmdet_results in mmdet_results_list:
        if isinstance(mmdet_results, tuple):
            det_results = mmdet_results[0]
        else:
            det_results = mmdet_results

        bboxes = det_results[cat_id]
        sorted_bbox = qsort_bbox_list(bboxes, only_max_arg)

        for bbox in sorted_bbox:
            person = {}
            person['bbox'] = bbox
            ret_list.append(person)
            if only_max_arg:
                break
    return ret_list


def get_area_of_bbox(bbox):
    """Get the area of a bbox_xyxy.

    Args:
        bbox (list):
            A list of [x1, y1, x2, y2] and x1<x2, y1<y2.

    Returns:
        float:
            Area of the bbox((y2-y1)*(x2-x1)).
    """
    return abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])


def qsort_bbox_list(input_list, only_max=False):
    """Sort a list of bboxes, by their area in pixel(W*H).

    Args:
        input_list (list):
            A list of bboxes. Each item is a list of (x1, y1, x2, y2)
        only_max (bool, optional):
            If True, only assure the max element at first place,
            others may not be well sorted.
            If False, return a well sorted descending list.
            Defaults to False.

    Returns:
        list:
            A sorted(maybe not so well) descending list.
    """
    if len(input_list) <= 1:
        return input_list
    else:
        bigger_list = []
        less_list = []
        anchor_index = int(len(input_list) / 2)
        anchor_bbox = input_list[anchor_index]
        anchor_area = get_area_of_bbox(anchor_bbox)
        for i in range(len(input_list)):
            if i == anchor_index:
                continue
            tmp_bbox = input_list[i]
            tmp_area = get_area_of_bbox(tmp_bbox)
            if tmp_area >= anchor_area:
                bigger_list.append(tmp_bbox)
            else:
                less_list.append(tmp_bbox)
        if only_max:
            return qsort_bbox_list(bigger_list) + \
                [anchor_bbox, ] + less_list
        else:
            return qsort_bbox_list(bigger_list) + \
                [anchor_bbox, ] + qsort_bbox_list(less_list)
