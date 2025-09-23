import argparse
import os
import shutil
import json
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from mmhuman3d.utils.ffmpeg_utils import images_to_video
import subprocess

def parse_args():
    parser = argparse.ArgumentParser('SMPLify tools')
    # path
    parser.add_argument('--subset', type=str,
        default='/mnt/cache/yinwanqi/01-project/zoehuman/data/data_list/extra.txt')
    parser.add_argument('--img', type=str,
        default='/mnt/cache/yinwanqi/01-project/zoehuman/data/RenBody')
    parser.add_argument('--msk', type=str,
        default='/mnt/cache/yinwanqi/01-project/zoehuman/data/RenBody_mask_test')
    parser.add_argument('--dst', type=str,
        # default='/mnt/cache/yinwanqi/01-project/zoehuman/data/RenBody_masked')
        default='/mnt/cache/yinwanqi/01-project/zoehuman/data/temp')
    parser.add_argument('--cam', type=str,
        default='25')
    parser.add_argument('--copy', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    # read txt
    # file = open(args.subset,'r')
    # file_success = open(args.successful,'r')
    # lines_raw = file.readlines()
    # lines_s_raw = file_success.readlines()
    # lines = [line.strip('\n') for line in lines_raw]
    # lines_s = [line.strip('\n') for line in lines_s_raw]
    # lines_s = [line.strip(' ') for line in lines_s]
    # lines_c = lines.copy()
    # lines_s_c = lines_s.copy()
    # lines.sort()
    # lines_s.sort()

    file = open(args.subset,'r')
    lines_raw = file.readlines()
    lines_sub = [line.strip('\n') for line in lines_raw]

    log = lines_sub.copy()
    for i, l in enumerate(lines_sub):
        # if i != 14:
        #     continue
        print(f"{i}/{len(lines_sub), l}***************")
        # src_root = "/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full"
        # dst_root = "/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full/mesh_test"
        video_name = l.replace("/","_")
        # src = os.path.join(args.src, l, 'mesh')
        # dst = os.path.join(args.dst, folder_name)
        # video_name = l.replace("/","_")
        # src = os.path.join(args.src, video_name+".mp4")
        # dst = os.path.join(args.dst, video_name+".mp4")
        # import pdb; pdb.set_trace()
        ### apose 
        for args.cam in range(60):
            print(args.cam)
            args.cam = f'{args.cam:02d}'
            image_dir = os.path.join(args.img, l, 'image', str(args.cam))
            mask_dir = os.path.join(args.msk, l, 'mask', str(args.cam))
            save_dir = os.path.join(args.dst, 'image_png', l, str(args.cam))
            video_dir = os.path.join(args.dst, 'video', f'{video_name}_{args.cam}.mp4')
            raw_video_dir = os.path.join(args.dst, 'raw_video', f'{video_name}_{args.cam}.mp4')

            os.makedirs(save_dir, exist_ok=True)

            image_list = os.listdir(image_dir)
            mask_list = os.listdir(mask_dir)

            image_list.sort()
            mask_list.sort()
            # import pdb; pdb.set_trace()

            for j, image_name in enumerate(tqdm(image_list)):
                # import pdb; pdb.set_trace()
                image_to_load = os.path.join(image_dir, image_name)
                mask_to_load = os.path.join(mask_dir, mask_list[j])
                image_name = image_name[:-3]  + 'png'
                # image_name = image_name[:-3]  + 'jpg'
                image_to_save = os.path.join(save_dir, image_name)

                image = cv2.imread(image_to_load)
                mask = cv2.imread(mask_to_load)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

                x, y = image.shape[:2]
                x_, y_ = mask.shape[:2]
                # import pdb; pdb.set_trace()
                if not (x == x_ and y == y_):
                    mask = Image.fromarray(mask)
                    mask = mask.resize((y, x))
                    mask = np.array(mask) 
                if mask.shape[2] == 3:
                    mask = mask[..., 0]
                
                mask_3 = image.copy()
                mask_3[..., 0] = mask
                mask_3[..., 1] = mask
                mask_3[..., 2] = mask

                # import pdb; pdb.set_trace()
                # image = np.multiply(image, mask_3/255) + 255 - mask_3 # jpg
                image[..., -1] = mask # png

                cv2.imwrite(image_to_save, image)

            # images_to_video(
            #     input_folder=save_dir,
            #     output_path=video_dir,
            #     remove_raw_file=False,
            #     img_format='%05d.jpg',
            #     fps=30)
            
            # images_to_video(
            #     input_folder=image_dir,
            #     output_path=raw_video_dir,
            #     remove_raw_file=False,
            #     img_format='%05d.jpg',
            #     fps=30)
            

        # save png to gif, mov
        # gif_command_1 = [
        #     'ffmpeg', 
        #     '-y',
        #     '-i', 
        #     f'{save_dir}/%05d.png', 
        #     '-vf',
        #     'palettegen',
        #     f'{video_dir}_palette.png',
        #     # '&&',
        #     # 'ffmpeg',
        #     # '-i',
        #     # f'{save_dir}/%05d.png',
        #     # '-i',
        #     # f'{video_dir}_palette.png',
        #     # '-lavfi', 
        #     # '"paletteuse"',
        #     # '-y',
        #     # f'{video_dir}.gif',
        # ]
        # gif_command_2 = [
        #     'ffmpeg',
        #     '-y',
        #     '-i',
        #     f'{save_dir}/%05d.png',
        #     '-i',
        #     f'{video_dir}_palette.png',
        #     '-lavfi', 
        #     'paletteuse',
        #     '-y',
        #     f'{video_dir}.gif',
        # ]

        # mov_command = [
        #     'ffmpeg',
        #     '-y',
        #     '-i',
        #     f'{save_dir}/%05d.jpg', 
        #     '-vcodec',
        #     'png',
        #     f'{video_dir}.mp4',
        # ]

        # mov_command = [
        #     'ffmpeg',
        #     '-y',
        #     '-i',
        #     f'{save_dir}/%05d.png', 
        #     '-vcodec',
        #     'png',
        #     f'{video_dir}.mov',
        # ]
        # # import pdb; pdb.set_trace()
        # print(f'Running \"{" ".join(gif_command_1)}\"')
        # subprocess.check_call(gif_command_1)
        # print(f'Running \"{" ".join(gif_command_2)}\"')
        # subprocess.check_call(gif_command_2)
        # os.remove(f'{video_dir}_palette.png')

        # import pdb; pdb.set_trace()
        # print(f'Running \"{" ".join(mov_command)}\"')
        # subprocess.check_call(mov_command)

        


    #     folder_name = l.replace("/","_")
    #     date, person_id = l.split('/')[:2]
    #     yf = l.split('/')[-1].split('_')[-2]
    #     name = l.split('/')[1][:-2]
    #     apose_person_folder = f'{name}_apose_{yf}'
    #     # print(f'apose folder: {apose_person_folder}')
    #     apose_output_root = '/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full_apose'
    #     src = os.path.join(apose_output_root,date,person_id,apose_person_folder, 'mesh')
    #     dst = os.path.join(args.dst, folder_name)
    #     print(f'Copying \n{src} to \n{dst}')
    #     date_m = int(l.split('_')[0][4:6])
    #     # if date_m < 7:
    #     #     continue
    #     if not os.path.exists(src):
    #         print("no mesh folder", l)
    #         continue
    #     if args.copy:
    #         if not os.path.exists(args.dst):
    #                 os.makedirs(args.dst)
    #         # shutil.copyfile(src, dst)
    #         shutil.copytree(src, dst)
    #         # if os.path.isfile(src):
    #         #     os.remove(src)
    #         # else:
    #         #     print(src)
    #     log.remove(l)

    # print(log)

    # print("#seq in subset: ",len(lines))
    # print("#seq in successful list: ",len(lines_s))
    # print("#seq in subset but not processed successfully: ", len(lines_c))
    # print("#seq remain in processed list: ", len(lines_s_c))

    # file.close()
    # file_success.close()
    
    # print("lack of these sequences, check reasons:")
    # for l in lines_c:
    #     print(l)
    
    # # print("extras:")
    # # for l in lines_s_c:
    # #     print(l)
    
    # jason
    # failure_list={}
    # with open(args.subset) as json_file:
    #     data = json.load(json_file)
    #     data.pop("definition")
    #     for category, data_list in data.items():
    #         log = data_list.copy()
    #         for sequence in data_list:
    #             video_name = sequence.replace("/","_")
    #             src = os.path.join(args.src, video_name+".mp4")
                
    #             dst_dir = os.path.join(args.dst, category)
    #             if not os.path.exists(dst_dir):
    #                 os.makedirs(dst_dir)
    #             dst = os.path.join(dst_dir, video_name+".mp4")
                
    #             print(f'Copying {src} to {dst}')
    #             if args.copy:
    #                 shutil.copyfile(src, dst)
    #             log.remove(sequence)
            
    #         failure_list[category] = log
        
    
    # print(failure_list)

    # failure_list={}
    # with open(args.subset) as json_file:
    #     data = json.load(json_file)
    #     data.pop("definition")
    #     for category, data_list in data.items():
    #         log = data_list.copy()
    #         for sequence in data_list:
    #             video_name = sequence.replace("/","_")
    #             src = os.path.join(args.src, video_name+".mp4")
                
    #             dst_dir = os.path.join(args.dst, category)
    #             if not os.path.exists(dst_dir):
    #                 os.makedirs(dst_dir)
    #             dst = os.path.join(dst_dir, video_name+".mp4")
                
    #             print(f'Copying {src} to {dst}')
    #             if args.copy:
    #                 shutil.copyfile(src, dst)
    #             log.remove(sequence)
            
    #         failure_list[category] = log
        
    
    # print(failure_list)

    
