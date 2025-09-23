import os
from subprocess import Popen

if __name__ == '__main__':

    nproc = 1
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    folder = '/mnt/lustre/yanglei/data/renbody/vis/'
    filelist = '/mnt/lustre/share_data/yanglei/renbody/filelist-20220412-all.txt'  # noqa: E501

    with open(filelist) as f:
        lst = [x for x in f.readlines()]
    tot = len(lst)
    num = int(tot / nproc) + 1

    print(f'#tot: {tot}, #num: {num}')
    for idx in range(nproc):
        filelist_idx = filelist[:-4] + f'_{idx}.txt'
        with open(filelist_idx, 'w') as of:
            beg = idx * num
            end = min((idx + 1) * num, tot)
            for x in lst[beg:end]:
                of.write(x)

        command = f'srun -u --mpi=pmi2 -p VI_MODEL_V100_32G_test \
                        --gres=gpu:1 -n1 --ntasks-per-node=1 \
                    python mocap/smplify_renbody/smplify3d.py \
                        --vis_smpl \
                        --kp3d_path {filelist_idx} \
                        --output_folder {folder} \
                        --model "smplx" \
                        --gender "neutral" \
                        --src_convention "human_data" \
                        --tgt_convention "openpose_118"'

        print(command)
        fdout = open(f'{log_dir}/{idx}.out', 'w')
        fderr = open(f'{log_dir}/{idx}.err', 'w')
        Popen(command, stdout=fdout, stderr=fderr, shell=True)
