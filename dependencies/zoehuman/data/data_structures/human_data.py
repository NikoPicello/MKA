from mmhuman3d.data.data_structures.human_data import HumanData as HumanData_mm  # noqa:F401, E501

_HumanData_SUPPORTED_KEYS = HumanData_mm.SUPPORTED_KEYS
_HumanData_SUPPORTED_KEYS.update({
    'smpl': {
        'type': dict,
        'slice_key': 'global_orient',
        'dim': 0
    },
    'smplh': {
        'type': dict,
        'slice_key': 'global_orient',
        'dim': 0
    },
    'smplx': {
        'type': dict,
        'slice_key': 'global_orient',
        'dim': 0
    }
})


class HumanData(HumanData_mm):
    logger = HumanData_mm.logger
    SUPPORTED_KEYS = _HumanData_SUPPORTED_KEYS
    WARNED_KEYS = HumanData_mm.WARNED_KEYS
