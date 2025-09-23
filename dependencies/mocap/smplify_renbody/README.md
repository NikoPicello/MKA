# SMPLify (RenBody)

## File structure

```
mocap/smplify_renbody/
 ├── body_models
 │   ├── body_model.py      # to use extra x_regressor defined in zju-easymocap, can be replaced by mmhuman3d/models/body_models/smplx.py
 │   ├── body_param.py      # only a load_model func
 │   ├── __init__.py
 │   └── lbs.py             # smpl-x lbs, can be replaced by official lbs (TODO: check)
 ├── config.py              # zju-config for using x_regressor
 ├── pipeline
 │   ├── basic.py           # **smpl_from_keypoints3d**
 │   ├── __init__.py
 │   └── weight.py          # weight for losses, similar mmhuman3d config
 ├── registrants
 │   ├── __init__.py
 │   ├── lbfgs.py           # can be replaced by official lbfgs (TODO: check)
 │   ├── loss.py            # loss functions
 │   ├── optimize.py        # FittingMonitor for early stop
 │   └── optimize_smpl.py   # optimizePose3D, optimizeShape
 └── smplify3d.py           # **main**
```
