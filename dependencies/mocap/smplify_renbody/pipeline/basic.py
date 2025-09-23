from mocap.smplify_renbody.config import CONFIG
from mocap.smplify_renbody.registrants import optimizePose3D, optimizeShape

from .weight import load_weight_pose, load_weight_shape


class Config:
    OPT_R = False
    OPT_T = False
    OPT_POSE = False
    OPT_SHAPE = False
    OPT_HAND = False
    OPT_EXPR = False
    verbose = False
    model = 'smpl'
    device = None

    def __init__(self, args=None) -> None:
        if args is not None:
            self.verbose = args.verbose
            self.model = args.model


def multi_stage_optimize(body_model,
                         params,
                         kp3ds,
                         weight={},
                         cfg=None,
                         scale=1):
    print('Optimize global RT')
    cfg.OPT_R = True
    cfg.OPT_T = True
    params = optimizePose3D(
        body_model, params, kp3ds, weight=weight, cfg=cfg, scale=scale)

    print('Optimize 3D Pose/{} frames'.format(kp3ds.shape[0]))
    cfg.OPT_POSE = True
    params = optimizePose3D(
        body_model, params, kp3ds, weight=weight, cfg=cfg, scale=scale)
    if cfg.model in ['smplh', 'smplx']:
        cfg.OPT_HAND = True
        params = optimizePose3D(
            body_model, params, kp3ds, weight=weight, cfg=cfg, scale=scale)
    if cfg.model == 'smplx':
        cfg.OPT_EXPR = True
        params = optimizePose3D(
            body_model, params, kp3ds, weight=weight, cfg=cfg, scale=scale)
    return params


def smpl_from_keypoints3d(body_model,
                          kp3ds,
                          config,
                          args,
                          weight_shape=None,
                          weight_pose=None,
                          opt_scale=False,
                          scale=1.0):
    model_type = body_model.model_type
    params_init = body_model.init_params(nFrames=1)
    if weight_shape is None:
        weight_shape = load_weight_shape(model_type, args.opts)
    if model_type in ['smpl', 'smplh', 'smplx']:
        # when use SMPL model, optimize the shape only with first 1-14 limbs,
        # don't use (nose, neck)
        print('[RB] Optimize shape!')
        # print('TODO: Actually, below has problem when kp3ds is not openpose, as kintree cannot match perfactly') # noqa: E501
        params_shape = optimizeShape(
            body_model,
            params_init,
            kp3ds,
            weight_loss=weight_shape,
            kintree=CONFIG['body15']['kintree'][1:],
            opt_scale=opt_scale,
            scale=scale)
    else:
        params_shape = optimizeShape(
            body_model,
            params_init,
            kp3ds,
            weight_loss=weight_shape,
            kintree=config['kintree'],
            opt_scale=opt_scale,
            scale=scale)
    # optimize 3D pose
    cfg = Config(args)
    cfg.device = body_model.device
    cfg.model_type = model_type
    params = body_model.init_params(
        nFrames=kp3ds.shape[0])  # 'poses': shape=(n, 87). 87=66+12+9
    params['shapes'] = params_shape['shapes'].copy()
    # if opt_scale:
    #     params['scale'] = params_shape['scale'].copy()
    # print('-------body scale after shape opti: '
    # , params['scale'].item(), '---------')
    if weight_pose is None:
        print('[RB] Load weight pose!')
        weight_pose = load_weight_pose(model_type, args.opts)
    # We divide this step to two functions,
    # because we can have different initialization method
    params = multi_stage_optimize(
        body_model, params, kp3ds, weight_pose, cfg, scale=scale)
    # print('-------body scale after multi opti: ',
    # params['scale'].item(), '---------')
    return params
