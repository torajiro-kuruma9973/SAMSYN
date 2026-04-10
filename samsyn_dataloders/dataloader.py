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
from samsyn_json_metadata import lasions_distribution
#from samsyn_json_metadata import count_ct_slices
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
            tr_data_path_list.append({"image_ct": img_name, "label": label_name})

        for i in range(len(val_paths)):
            img_name = args.data_root + val_paths[i]
            label_name = img_name.replace("data/", "labels/")
            val_data_path_list.append({"image_ct": img_name, "label": label_name})
    
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
            LoadImaged(keys=['image_ct', 'label']),
            AddChanneld(keys=['image_ct', 'label']),
            Orientationd(keys=['image_ct', 'label'], axcodes="RAS"),
            ForegroundNormalization(keys=['image_ct']),
            PermuteTransform(keys=['image_ct', 'label'], dims=(3,0,1,2)),
            ]
            )

        self.transform_2d = transforms.Compose(
                [
                    Resize(keys=['image_ct', 'label'], target_size=(self.image_size, self.image_size), num_class=len(self.class_dict)),  #
                    transforms.ToTensord(keys=['image_ct', 'label']),
                    Normalization(keys=['image_ct', 'label']),
                ])

        self.transform_2d_label = transforms.Compose(
                [
                    Resize(keys=["label"], target_size=(self.image_size, self.image_size), num_class=len(self.class_dict)),  #
                    transforms.ToTensord(keys=["label"]),
                ])
        
        self.num_objs = args.num_objs

        # get lasions coords info from json
        self.lasions_coords_info_dict = lasions_distribution.process_and_map_json_with_coords(args.lasion_ct_pix_json, args.rename_json)
        self.ct_slices_counts = utils.load_json_to_dict(samsyn_cfg.ct_slice_counts_json)
        self.nii_idx_with_prompts_coords = utils.load_json_to_dict(samsyn_cfg.nii_idx_with_prompts_coords_json)
        self.interval_info = utils.load_json_to_dict(samsyn_cfg.interval_info)
   
    def __len__(self):
        return len(self.data_paths)


    def normalize_label(self, label):
        label[label != 0] = 1
        return label

    
    def _generate_slices(self, case_name, ct_slice_counts):
        interval_info_dict = utils.load_json_to_dict(samsyn_cfg.interval_info)
        interval_list = interval_info_dict[case_name]
        if len(interval_list) > samsyn_cfg.num_intervals:
            intervals = random.sample(interval_list, samsyn_cfg.num_intervals)
            starting_slices = [x[0] for x in intervals]
            end_slices = [x[1] for x in intervals]
        elif len(interval_list) == 0: # no freground, pick bg points.
            starting_slices = random.sample(range(ct_slice_counts), samsyn_cfg.num_intervals)
            end_slices = [x + samsyn_cfg.interval_thickness for x in starting_slices]
        else:
            starting_slices = [x[0] for x in interval_list]
            end_slices = [x[1] for x in interval_list]
        return starting_slices, end_slices


    def __getitem__(self, index):
        image3d_ct = self.data_paths[index]['image_ct']
        label3d_path = self.data_paths[index]['label']
        case_name = image3d_ct.split('/')[-1].split('\\')[-1]
        case_name = case_name.split('.')[0]
        case_idx = int(case_name.split('.')[0])
        print("############################")
        print(image3d_ct)
        print(label3d_path)
        print(case_name)
        print(index)
        print("############################")
        
        item_load = self.data3d_loader({'image_ct': image3d_ct, 'label': label3d_path})
        
        if item_load['image_ct'].shape != item_load['label'].shape:
            print(f"{image3d_ct} shape mismatch, skipping...")
            return self.__getitem__(np.random.randint(self.__len__()))

        item_load['label'] = self.normalize_label(item_load['label'])

        # here we should use lesions coords info instead of labels.
        lasions_info = self.lasions_coords_info_dict[case_idx]
        lasions_slice_info = list(lasions_info.keys())
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"foregrounds in slices: {lasions_slice_info}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        ct_slice_counts = self.ct_slices_counts[case_name]

        if len(lasions_slice_info) == 0: # no lasions are detected.
            first_nonzero_slice = 0
            last_nonzero_slice = ct_slice_counts
        else:
            first_nonzero_slice = min(lasions_slice_info)
            last_nonzero_slice = max(lasions_slice_info)
        
        last_nonzero_slice = last_nonzero_slice+self.slice_length if last_nonzero_slice+self.slice_length < ct_slice_counts else ct_slice_counts
        self.image3d_ct = item_load['image_ct'][first_nonzero_slice:last_nonzero_slice]
        self.label3d = item_load['label'][first_nonzero_slice:last_nonzero_slice]
        
        #num_slice = self.label3d.shape[0]
        (h,w) = self.label3d.shape[2:]
        print(f"H, W: {h}, {w}")

        starting_slices, end_slices = self._generate_slices(case_name, ct_slice_counts)
        print("AAAAAAAAAAAAAAAAAAAAAAA")
        print(starting_slices)
        print(end_slices)
        print("AAAAAAAAAAAAAAAAAAAAAAA")
        output_dict = {"obj_to_class": self.class_dict, "batch_input": []}

        points_info = self.lasions_coords_info_dict[case_idx]
        print("BBBBBBBBBBBBBBBBBBBBBBBB")
        print(case_idx)
        print(points_info)
        print("BBBBBBBBBBBBBBBBBBBBBBBB")
        #print(points_info)
        for star_slice, end_slice in zip(starting_slices, end_slices):
            output = self.process_3d_slices_with_prompts(star_slice, end_slice, points_info, h, w)
            #print(output)
            
            # if output is not None:
            #     output_dict["batch_input"].append(output)
        
        if len(output_dict['batch_input']) == 0:
            return self.__getitem__(np.random.randint(self.__len__()))


        output_dict['start_end'] = (first_nonzero_slice, last_nonzero_slice)
        output_dict['ori_label'] = item_load['label']
        output_dict['case_name'] = case_name
        
        return output_dict

    def process_3d_slices_with_prompts(self, starting_slice, end_slice, points_info, h, w):
        start_slice = starting_slice
        if len(points_info) == 0: # slect bg points
            y = random.sample(h, samsyn_cfg.points_num)
            x = random.sample(w, samsyn_cfg.points_num)
            point_coords = [list(pair) for pair in zip(y, x)]
            point_labels = [0] * samsyn_cfg.points_num
        else:
            points = points_info[start_slice]

            point_coords, point_labels = get_points_from_mask(points, samsyn_cfg.points_num, h, w) 
            #bboxes = get_bboxes_from_mask(label_2d["label"], offset=5) 

        start_objs = np.unique(points)
        
        if len(start_objs) < 2:  # 无足够的对象时直接返回
            return None
        
        start_objs = (start_objs[start_objs != 0] - 1).astype(np.uint8)  # 移除背景类别并调整索引
        num_obj = min(len(start_objs), self.num_objs) 
        select_obj = random.sample(list(start_objs), num_obj)
 
        image_ct = torch.zeros(end_slice-starting_slice, 3, self.image_size, self.image_size)
        all_label = {obj: [] for obj in select_obj}
        all_prompt = {obj: {'point_coords': {}, 'point_labels': {}, 'bboxes': {}} for obj in select_obj}
        
        for i, slice_index in enumerate(range(starting_slice, end_slice)):
            label2d = self.label3d[slice_index]
            image2d_ct = self.image3d_ct[slice_index]
            
            item_2d = self.transform_2d({"image_ct":image2d_ct, "label":label2d})
            image_ct[i,...] = item_2d["image_ct"]
            
            if item_2d["label"].sum() == 0:
                print("May cause some errors......................")
                return None

            for idx, obj in enumerate(select_obj):
                all_label[obj].append(item_2d["label"][obj])
                if slice_index == start_slice:
                    all_prompt[obj]['point_coords'][i] = point_coords[obj].numpy()
                    all_prompt[obj]['point_labels'][i] = point_labels[obj].numpy()

        for obj in select_obj:
            all_label[obj] = torch.stack(all_label[obj], dim=0)

        return {'image_ct':image_ct, 'label': all_label, 'prompt':all_prompt}



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