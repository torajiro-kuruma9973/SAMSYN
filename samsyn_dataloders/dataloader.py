import os
import numpy as np
import samsyn_cfg
import torch
from PIL import Image, ImageDraw  
from torch.utils.data import Dataset
import json 
from monai import data, transforms
from sklearn.model_selection import train_test_split
from samsyn_dataloders.data_utils import (  # 
    Resize, 
    PermuteTransform, 
    ForegroundNormalization,
    LongestSidePadding, 
    Normalization, 
    get_points_from_mask, 
    get_bboxes_from_mask
    )
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    )
import random
import cv2
from torch.utils.data.distributed import DistributedSampler
from samsyn_json_metadata import utils


def sample_collate_fn(batch):
    assert len(batch) == 1, 'Please set batch size to 1 when testing mode'
    
    batch_image_flair = []
    batch_image_t1 = []
    pre_interval_obj_label = {}
    pre_interval_obj_prompt = {}
    
    for interval, samples in enumerate(batch[0]["batch_input"]):
        batch_image_flair.append(samples["image_flair"])
        batch_image_t1.append(samples["image_t1"])
        pre_interval_obj_label[interval] = samples["label"]
        pre_interval_obj_prompt[interval] = samples["prompt"]
    image_flair = torch.stack(batch_image_flair, dim=0)
    image_t1 = torch.stack(batch_image_t1, dim=0)

    return {
        "pre_interval_image_flair":image_flair,
        "pre_interval_image_t1":image_t1,
        "pre_interval_obj_label":pre_interval_obj_label,
        "pre_interval_obj_prompt":pre_interval_obj_prompt,
        'obj_to_class': batch[0]["obj_to_class"],
        'ori_label': batch[0].get('ori_label', None),
        'start_end': batch[0].get('start_end', None),
        'case_name': batch[0].get('case_name', None)
    }


class dataset_3d(Dataset):
    def __init__(self, args, mode):
        # json_path = os.path.join(args.data_root, 'new_dataset.json')
        # with open(json_path, 'r') as file:  
        #      self.data_json = json.load(file)  
        # if args.num_intervals != 0:
        #     self.num_intervals = args.num_intervals
        # else: # that means the intervals are not same, each value should be calculated case by case
        self.num_intervals = args.num_intervals
        # self.class_dict = self.data_json['labels']

        self.class_dict = {"0": "background", "1": "foreground"}

        if "0" in self.class_dict:
            del self.class_dict["0"]

        updated_class_dict = {}
        for key, value in self.class_dict.items():
            updated_class_dict[int(key) - 1] = value

        self.class_dict = {k: v for k, v in sorted(updated_class_dict.items())} 
        
        total_files_names = [name for name in os.listdir(args.data_root) if os.path.isfile(os.path.join(args.data_root, name))]
        #train_paths, temp_paths = train_test_split(self.data_json['training'], test_size=0.4, random_state=42)  # 60% 训练，40% 临时集
        train_paths, temp_paths = train_test_split(total_files_names, test_size=0.2, random_state=42)
        #val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)  # 20% 验证，20% 测试
        val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42) 
        

        tr_data_path_list = []
        val_data_path_list = []
        for i in range(len(train_paths)):
            img_name = args.data_root + train_paths[i]
            label_name = img_name.replace("data/", "labels/")
            seg_name = img_name.replace("data/", "segs/")
            tr_data_path_list.append({"image_data": img_name, "label": label_name, "seg": seg_name})

        for i in range(len(val_paths)):
            img_name = args.data_root + val_paths[i]
            label_name = img_name.replace("data/", "labels/")
            seg_name = img_name.replace("data/", "segs/")
            val_data_path_list.append({"image_data": img_name, "label": label_name, "seg": seg_name})
    
        if mode == 'training':
            self.data_paths = tr_data_path_list
            self.slice_length = args.slice_length
        elif mode == 'val':
            self.data_paths = val_data_path_list
            self.slice_length = -1
        else:
            print("Dont do test in this file!!!")
            assert(1 < 0)
            #self.data_paths = test_paths
            #self.slice_length = -1

        self.data_root = args.data_root
        self.image_size = args.image_size

        self.data3d_loader = Compose([
            LoadImaged(keys=['image_data', 'label', 'seg']),
            AddChanneld(keys=['image_data', 'label', 'seg']),
            Orientationd(keys=['image_data', 'label', 'seg'], axcodes="RAS"),
            #ForegroundNormalization(keys=['seg']),
            PermuteTransform(keys=['image_data', 'label', 'seg'], dims=(3,0,1,2)),
            ]
            )

        self.transform_2d = transforms.Compose(
                [
                    Resize(keys=['image_data', 'label'], target_size=(self.image_size, self.image_size), num_class=len(self.class_dict)),  #
                    transforms.ToTensord(keys=['image_data', 'label']),
                    # Normalization(keys=['image_data', 'label', 'seg']),
                ])

        self.transform_2d_seg = transforms.Compose(
                [
                    Resize(keys=["seg"], target_size=(self.image_size, self.image_size), num_class=len(self.class_dict)),  #
                    transforms.ToTensord(keys=["seg"]),
                ])
        
        self.num_objs = args.num_objs

        # get lasions coords info from json
        self.prompts_info = utils.read_json_to_dict("samsyn_json_metadata/seg_points_info_with_idx.json")
   
    def __len__(self):
        return len(self.data_paths)


    def normalize_label(self, label):
        label[label != 0] = 1
        return label

    
    def _generate_slices(self, slice_info, total_slices_num):
        int_slice_info = [int(x) for x in slice_info]
        assert len(int_slice_info) > 0

        if len(int_slice_info) > samsyn_cfg.num_intervals:
            slices = random.sample(int_slice_info, samsyn_cfg.num_intervals)
            starting_slices = [x for x in slices]
            # for in case out of bound:
            end_slices = [min(x + samsyn_cfg.interval_thickness, total_slices_num) for x in starting_slices]
        # must not run
        elif len(int_slice_info) == 0: # no freground, we still select some intevals which dont contain forefround.
            starting_slices = random.sample(range(total_slices_num), samsyn_cfg.num_intervals)
            end_slices = [min(x + samsyn_cfg.interval_thickness, total_slices_num) for x in starting_slices]
        else:
            starting_slices = [x for x in int_slice_info]
            end_slices = [min(x + samsyn_cfg.interval_thickness, total_slices_num) for x in starting_slices]
        starting_slices.sort()
        end_slices.sort()
        return starting_slices, end_slices


    def __getitem__(self, index):
        image3d_data_path = self.data_paths[index]['image_data']
        label3d_path = self.data_paths[index]['label']
        case_name = image3d_data_path.split('/')[-1].split('\\')[-1]
        case_name = case_name.split('.')[0]
        seg_path = self.data_paths[index]['seg']
        print("############################")
        print(image3d_data_path)
        print(label3d_path)
        print(seg_path)
        #print(index)
        print("############################")
        
        item_load = self.data3d_loader({'image_data': image3d_data_path, 'label': label3d_path, 'seg': seg_path})
        if item_load['image_data'].shape != item_load['label'].shape:
            print(f"{image3d_data_path} shape mismatch, skipping...")
            return self.__getitem__(np.random.randint(self.__len__()))
        
        item_load['seg'] = self.normalize_label(item_load['seg'])

        total_slice_num = item_load['image_data'].shape[0]
        # here we should use lesions coords info instead of labels.
        lasions_info = self.prompts_info[case_name]
        lasions_slice_info = list(lasions_info.keys())
        starting_slices, end_slices = self._generate_slices(lasions_slice_info, total_slice_num)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(case_name)
        print(total_slice_num)
        print(f"start: {starting_slices}")
        print(f"end: {end_slices}")
        #print(f"lasions_info[{case_name}] = {lasions_info}")

        #first_nonzero_slice = starting_slices[0]
        first_nonzero_slice = 0 # load all pics
        last_nonzero_slice = end_slices[-1]
        print(f"first_nonzero_slice: {first_nonzero_slice}")
        print(f"last_nonzero_slice: {last_nonzero_slice}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
        self.image3d_data = item_load['image_data'][first_nonzero_slice:last_nonzero_slice]
        self.label3d = item_load['label'][first_nonzero_slice:last_nonzero_slice]
        self.seg3d = item_load['seg']
        (h,w) = self.label3d.shape[2:]
        print(f"H, W: {h}, {w}")

        output_dict = {"obj_to_class": self.class_dict, "batch_input": []}
        
        for star_slice, end_slice in zip(starting_slices, end_slices):
            output = self.process_3d_slices_with_prompts(case_name, star_slice, end_slice, lasions_info, h, w)
            #print(output)
            
            # if output is not None:
            #     output_dict["batch_input"].append(output)
        
        if len(output_dict['batch_input']) == 0:
            return self.__getitem__(np.random.randint(self.__len__()))


        output_dict['start_end'] = (first_nonzero_slice, last_nonzero_slice)
        output_dict['ori_label'] = item_load['label']
        output_dict['case_name'] = case_name
        
        return output_dict

    def process_3d_slices_with_prompts(self, case_name, starting_slice, end_slice, lasions_info, h, w):
        start_slice = starting_slice
        #print(lasions_info)
        start_slice_str = str(start_slice)
        seg_idx = lasions_info[start_slice_str] # seg frame idx
        seg_frame = self.seg3d[seg_idx]
        
        assert len(list(lasions_info.keys())) > 0 # the folders which dont have foreground have been deleted.
        
        seg_2d = self.transform_2d_seg({"seg":seg_frame}) #[1, 1024, 1024] already.
        seg_2d_tensor = seg_2d['seg'][0]
        points = torch.nonzero(seg_2d_tensor).tolist()
        objs = seg_2d_tensor[seg_2d_tensor != 0].tolist()
        coords = [sublist[::-1] for sublist in points]  # [x,y] --> [y,x]
        prompts = list(zip(coords, objs))
        point_coords, point_labels = get_points_from_mask(prompts, samsyn_cfg.points_num, h, w) 
    

        start_objs = np.unique(seg_frame)
        
        if len(start_objs) < 2:  # 无足够的对象时直接返回
            return None
        
        start_objs = (start_objs[start_objs != 0] - 1).astype(np.uint8)  # 移除背景类别并调整索引
        num_obj = min(len(start_objs), self.num_objs) 
        select_obj = random.sample(list(start_objs), num_obj)
 
        image_data = torch.zeros(end_slice-starting_slice, 3, self.image_size, self.image_size)
        all_label = {obj: [] for obj in select_obj}
        all_prompt = {obj: {'point_coords': {}, 'point_labels': {}} for obj in select_obj}
        
        for i, slice_index in enumerate(range(starting_slice, end_slice)):
            
            label2d = self.label3d[slice_index]
            image2d_data = self.image3d_data[slice_index]

            if image2d_data.shape[0] != 3: # gray --> jpg by simply copying
                image2d_data = np.tile(image2d_data, (3, 1, 1)) 
                label2d = np.tile(label2d, (3, 1, 1))
            
            item_2d = self.transform_2d({"image_data":image2d_data, "label":label2d})
            image_data[i,...] = item_2d["image_data"]
            
            # if item_2d["label"].sum() == 0:
            #     print("May cause some errors......................")
            #     return None

            for idx, obj in enumerate(select_obj):
                all_label[obj].append(item_2d["label"][obj])
                if slice_index == start_slice:
                    all_prompt[obj]['point_coords'][i] = point_coords[obj].numpy()
                    all_prompt[obj]['point_labels'][i] = point_labels[obj].numpy()

        for obj in select_obj:
            all_label[obj] = torch.stack(all_label[obj], dim=0)

        return {'image_data':image_data, 'label': all_label, 'prompt':all_prompt}



def get_dataset_3d(args, mode):
    train_data = dataset_3d(args, mode)
    print(f'{mode} dataset: {len(train_data)}')
    train_sampler = DistributedSampler(train_data, shuffle=False) if args.dist else None
    train_loader = data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=sample_collate_fn,
        )
    return train_loader
    

if __name__ == "__main__":
    from torchvision import transforms as torch_trans 
    import argparse
    import time
    # dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    def set_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_root", type = str, default='datasets/BraTS2020/FLAIR')
        parser.add_argument('--image_size', type=int, default=1024)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--mode', type = str, default='test')
        parser.add_argument('--slice_length', type=int, default=8)
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--num_intervals', type=int, default=4)
        parser.add_argument('--num_objs', type=int, default=1)
        parser.add_argument('--dist', dest='dist', type=bool, default=False, help='distributed training or not')
        args = parser.parse_args()
        return args
    
    args = set_parse()

    train_loader = get_dataset_3d(args, 'test')
    start_time = time.time()  
    for idx, batch_input in enumerate(train_loader):
        print(batch_input['ori_shape'], batch_input['start_end'])
        obj_to_class = batch_input["obj_to_class"]
        interval_prompts = batch_input["pre_interval_obj_prompt"]
        interval_images_flair, interval_images_t1, interval_labels = batch_input["pre_interval_image_flair"], batch_input["pre_interval_image_t1"], batch_input["pre_interval_obj_label"]
    
        for interval in range(interval_images_flair.shape[0]):
            images_flair = interval_images_flair[interval]
            images_t1 = interval_images_t1[interval]
            labels = interval_labels[interval]
            prompts = interval_prompts[interval]
            #每个label的obj
            print(list(labels.keys()))
            print(images_flair.shape, images_t1.shape, labels.keys())
            for obj in list(labels.keys()):
                obj_label = labels[obj]
                obj_point_coord = prompts[obj]["point_coords"]
                obj_point_label = prompts[obj]["point_labels"]

    end_time = time.time()  
    # 计算并打印处理时间  
    print(f"Batch {idx} processing time: {end_time - start_time:.4f} seconds")