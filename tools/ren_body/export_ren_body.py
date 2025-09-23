import os
import shutil
import sys

import numpy as np
from tqdm import trange

if __name__ == '__main__':

    zoehumandir = sys.argv[1]
    renbodydir = sys.argv[2]
    tardir = sys.argv[3]

    paramdir = [f for f in os.listdir(zoehumandir) if 'smpl' in f][0]
    paramname = f'{zoehumandir}/{paramdir}/human_data_tri_smplx.npz'
    smpldir = f'{zoehumandir}/mesh'

    names = sorted(os.listdir(f'{renbodydir}/image/00'))
    smpls = sorted(os.listdir(smpldir))

    os.makedirs(f'{tardir}/param', exist_ok=True)
    os.makedirs(f'{tardir}/smpl', exist_ok=True)

    params = np.load(paramname)
    keys = list(set(list(params.keys())) - set(['scale']))
    out_params = {}
    for i in trange(len(names)):
        new_param = {}
        for key in keys:
            if len(params[key]) == 150:
                new_param[key] = params[key][i].copy()
            else:
                new_param[key] = params[key].copy()
        for key in keys:
            if key not in ['betas', 'left_hand_pose', 'right_hand_pose']:
                new_param[key] = new_param[key][np.newaxis, ...]
        out_params['smplx'] = new_param
        out_params['smplx_scale'] = params['scale']
        np.save(
            os.path.join(tardir, 'param', names[i][:-4] + '.npy'), out_params)
        srcdir = os.path.join(smpldir, smpls[i])
        objdir = os.path.join(tardir, 'smpl', names[i][:-4] + '.obj')
        shutil.copy(srcdir, objdir)
