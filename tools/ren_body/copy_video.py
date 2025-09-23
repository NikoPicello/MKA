import argparse
import os
import shutil
import json

def parse_args():
    parser = argparse.ArgumentParser('SMPLify tools')
    # path
    parser.add_argument('--subset', type=str, required=True)
    parser.add_argument('--successful', type=str)
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
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
        print(f"{i}/{len(lines_sub)}***************")
        # src_root = "/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full"
        # dst_root = "/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full/mesh_test"
        # folder_name = l.replace("/","_")
        # src = os.path.join(args.src, l, 'mesh')
        # dst = os.path.join(args.dst, folder_name)
        # video_name = l.replace("/","_")
        # src = os.path.join(args.src, video_name+".mp4")
        # dst = os.path.join(args.dst, video_name+".mp4")
        
        ### apose 
        folder_name = l.replace("/","_")
        date, person_id = l.split('/')[:2]
        yf = l.split('/')[-1].split('_')[-2]
        name = l.split('/')[1][:-2]
        apose_person_folder = f'{name}_apose_{yf}'
        # print(f'apose folder: {apose_person_folder}')
        apose_output_root = '/mnt/cache/yinwanqi/01-project/zoehuman/data/zoehuman_ren_body_full_apose'
        src = os.path.join(apose_output_root,date,person_id,apose_person_folder, 'mesh')
        dst = os.path.join(args.dst, folder_name)
        print(f'Copying \n{src} to \n{dst}')
        date_m = int(l.split('_')[0][4:6])
        # if date_m < 7:
        #     continue
        if not os.path.exists(src):
            print("no mesh folder", l)
            continue
        if args.copy:
            if not os.path.exists(args.dst):
                    os.makedirs(args.dst)
            # shutil.copyfile(src, dst)
            shutil.copytree(src, dst)
            # if os.path.isfile(src):
            #     os.remove(src)
            # else:
            #     print(src)
        log.remove(l)

    print(log)

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

    
