import random
import numpy as np
import samsyn_cfg
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from monai import data, transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from skimage.measure import label, regionprops

class Resize(transforms.Transform):
    def __init__(self, keys, target_size, num_class):
        self.keys = keys
        self.target_size = target_size
        self.num_class = num_class
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if 'image' in key or 'label' in key:
                image = d[key]
                image = np.transpose(image, (1,2,0))
                image = to_pil_image(image)
                image = resize(image, self.target_size, interpolation=InterpolationMode.NEAREST)
                image_np = np.array(image)
                d[key] =  np.transpose(image_np, (2,0,1))
            elif key == 'seg':
                label = d[key]
                resized_labels = np.zeros((self.num_class, self.target_size[0], self.target_size[1]))
                uni_label = np.sort(np.unique(label))
               
                if 0 in uni_label:
                    uni_label = uni_label[1:]
    
                for i in uni_label:
                    label_modified = np.where(label[0] == i, 1, 0).astype(np.uint8)
                    pil_label = to_pil_image(label_modified)
                    resized_label = resize(pil_label, self.target_size, interpolation=InterpolationMode.NEAREST)
                    resized_labels[int(i-1)] = np.array(resized_label)
                d[key] = resized_labels
            else:
                raise ValueError(f"Unsupported image shape {image.shape} for key {key}. Expected (3, H, W) or (H, W, 3).") 
        return d


class Resize_2d(transforms.Transform):
    def __init__(self, keys, target_size):
        self.keys = keys
        self.target_size = target_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if len(d[key].shape) == 4:
                label = d[key]
                resized_labels = np.zeros((label.shape[0], self.target_size[0], self.target_size[1]))
                for i in range(label.shape[0]):
                    pil_label = to_pil_image(label[i])
                    resized_label = resize(pil_label, self.target_size, interpolation=InterpolationMode.NEAREST)
                    resized_labels[i] = np.array(resized_label)
                d[key] = resized_labels
            else:
                image = to_pil_image(d[key])
                d[key] = resize(image, self.target_size, interpolation=InterpolationMode.NEAREST)
                d[key] = np.array(d[key])
            
            if len(d[key].shape) == 2:
                d[key] = d[key][np.newaxis, ...]
        return d

class PermuteTransform(transforms.Transform):
    def __init__(self, keys, dims):
        self.dims = dims
        self.keys = keys
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.transpose(d[key], self.dims)
        return d

class LongestSidePadding(transforms.Transform):
    def __init__(self, keys, input_size):
        self.keys = keys
        self.input_size = input_size
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            h, w = d[key].shape[-2:]
            padh = self.input_size - h
            padw = self.input_size - w
            d[key] = F.pad(d[key], (0, padw, 0, padh))
        return d

class Normalization(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
        pixel_mean=(0.485, 0.456, 0.406),
        pixel_std=(0.229, 0.224, 0.225),
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = d[key] / 255.0
            d[key] = (d[key] - self.pixel_mean) / self.pixel_std
        return d
    
class ForegroundNormalization(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            d[key] = self.normalize(d[key])
        return d
    
    def normalize(self, ct_narray):
        print("##########################XXXXXXXXXXXXXXXXXXX")
        print(ct_narray.shape)
        a_transposed = np.transpose(ct_narray, (3, 0, 1, 2))
        coords_2d = np.argwhere(a_transposed[0][0]).tolist()
        print(len(coords_2d))
        print("##########################XXXXXXXXXXXXXXXXXXX")
        ct_voxel_ndarray = ct_narray.copy()
        ct_voxel_ndarray = ct_voxel_ndarray.flatten()
        thred = np.mean(ct_voxel_ndarray)
        voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
        upper_bound = np.percentile(voxel_filtered, 99.95)
        lower_bound = np.percentile(voxel_filtered, 00.05)
        ### transform ###
        ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
        ct_narray = (ct_narray - lower_bound) / (upper_bound - lower_bound)

        # mean = np.mean(voxel_filtered)
        # std = np.std(voxel_filtered)
        # ct_narray = (ct_narray - mean) / max(std, 1e-8)

        # ct_narray = (ct_narray * 255).astype(np.uint8) #不加对比度高点
        print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
        print(ct_narray.shape)
        a_transposed = np.transpose(ct_narray, (3, 0, 1, 2))
        coords_2d = np.argwhere(a_transposed[0][0]).tolist()
        print(len(coords_2d))
        print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
        return ct_narray 


def get_bboxes_from_mask(masks, offset=0):
    if masks.size(1) == 1:
        masks = masks.squeeze(1)
    B, H, W = masks.shape
    bounding_boxes = []
    for i in range(B):
        mask = masks[i]
        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
        
        if len(y_coords) == 0 or len(x_coords) == 0:
            bounding_boxes.append((0, 0, 0, 0))
        else:
            y0, y1 = y_coords.min().item(), y_coords.max().item()
            x0, x1 = x_coords.min().item(), x_coords.max().item()

            if offset > 0:
                y0 = max(0, y0 + torch.randint(-offset, offset + 1, (1,)).item())
                y1 = min(W - 1, y1 + torch.randint(-offset, offset + 1, (1,)).item())
                x0 = max(0, x0 + torch.randint(-offset, offset + 1, (1,)).item())
                x1 = min(H - 1, x1 + torch.randint(-offset, offset + 1, (1,)).item())

            bounding_boxes.append((x0, y0, x1, y1))
    return torch.tensor(bounding_boxes, dtype=torch.float).unsqueeze(1)


def get_random_points_safe(h, w, n, exclude_list):
    """
    通过 1D 索引映射，安全生成 n 个不在 exclude_list 中的 (y, x) 坐标对。
    """
    total_points = h * w
    
    # 1. 将黑名单的 2D 坐标转换为 1D 索引，并存入 Set 以获得 O(1) 查询速度
    exclude_set = {y * w + x for y, x in exclude_list}
    
    # 检查是否有足够的剩余空间
    if n > total_points - len(exclude_set):
        raise ValueError(f"请求 {n} 个点，但可用空间只有 {total_points - len(exclude_set)} 个！")
        
    # 2. 生成所有可用的 1D 索引池
    # 先用 set 减法剔除黑名单，再转回 list 供抽样
    available_indices = list(set(range(total_points)) - exclude_set)
    
    # 3. 无放回随机抽样 n 个点
    sampled_indices = random.sample(available_indices, n)
    
    # 4. 将抽中的 1D 索引还原为 2D 坐标 (y, x)
    result = [(idx // w, idx % w) for idx in sampled_indices]
    
    return result


def get_points_from_mask(prompts, points_num, h, w):
    
    if len(prompts) < points_num:
        fg = len(prompts)
    else:
        fg = points_num // 2
    
    bg = points_num - fg

    fg_list = random.sample(prompts, fg)
    fg_coords = [x[0] for x in fg_list]
    fg_obj_ids = [x[1] for x in fg_list]

    all_fg_coords = [x[0] for x in fg_list]
    
    bg_coords = get_random_points_safe(h, w, bg, all_fg_coords)
    bg_obj_ids = [0] * bg
    selected_coordinates = all_fg_coords + bg_coords
    selected_coordlabels = fg_obj_ids + bg_obj_ids

    print("@@@@@@@@@@@@@@@@@@@@@@@@")
    print(selected_coordinates)
    print(selected_coordlabels)
    print("@@@@@@@@@@@@@@@@@@@@@@@@")

    return torch.tensor(selected_coordinates), torch.tensor(selected_coordlabels)


