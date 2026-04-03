import os
from sys import breakpointhook
import pydicom
import numpy as np
import SimpleITK as sitk
from collections import defaultdict
import seg_process as sp

# mapping the lasions pos to CT pix space.
# space_dict: return from "get_frontground_from_seg"

def map_physical_to_ct_pixels(space_dict, ct_dir, z_tolerance=2.0):

    # ---------------------------------------------------------
    # 第一步：构建 CT 序列的空间元数据索引库
    # ---------------------------------------------------------
    ct_slices = []
    
    for f in os.listdir(ct_dir):
        if not f.lower().endswith('.dcm'):
            continue
            
        path = os.path.join(ct_dir, f)
        # stop_before_pixels=True 仅读取元数据，扫描极快
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        
        if hasattr(ds, 'ImagePositionPatient'):
            origin = np.array([float(v) for v in ds.ImagePositionPatient])
            spacing = [float(v) for v in ds.PixelSpacing] # [dy, dx]
            orientation = [float(v) for v in ds.ImageOrientationPatient]
            
            ct_slices.append({
                "name": f,
                "z": origin[2], # Z 轴物理高度 (Slice Location)
                "origin": origin,
                "dy": spacing[0],
                "dx": spacing[1],
                "row_dir": np.array(orientation[:3]), # X 轴方向向量
                "col_dir": np.array(orientation[3:]), # Y 轴方向向量
                "rows": ds.Rows,
                "cols": ds.Columns
            })
            
    if not ct_slices:
        raise ValueError(f"在 {ct_dir} 中未找到带有空间信息的 CT 文件。")

    # ---------------------------------------------------------
    # 第二步：遍历空间坐标字典，执行逆向空间投影
    # ---------------------------------------------------------
    results = {}
    
    for key, points in space_dict.items():
        for pt, obj_id in points:
            wx, wy, wz = pt
            if(obj_id != 1):
                print("Wow! A different obj ID {obj_id} !!")
            # 1. 寻找 Z 轴高度最匹配的 CT 切片
            closest_ct = None
            min_dist = float('inf')
            
            for ct in ct_slices:
                dist = abs(ct["z"] - wz)
                if dist < min_dist:
                    min_dist = dist
                    closest_ct = ct
                    
            # 检查是否在允许的物理层厚容差范围内
            if closest_ct and min_dist <= z_tolerance:
                ct_name = closest_ct["name"]
                
                # 使用 set 来存储，自动过滤掉四舍五入后重叠的同一像素点
                if ct_name not in results:
                    results[ct_name] = set() 
                
                # 提取该 CT 的专属空间参数
                origin = closest_ct["origin"]
                dx, dy = closest_ct["dx"], closest_ct["dy"]
                row_dir, col_dir = closest_ct["row_dir"], closest_ct["col_dir"]
                
                # 2. 向量点乘投影算像素
                # 计算病灶物理点相对于这张 CT 原点的三维向量
                vec_from_origin = np.array([wx, wy, wz]) - origin
                
                # 分别将该向量投影到 CT 的列方向(X)和行方向(Y)上，并除以间距
                px = int(round(np.dot(vec_from_origin, row_dir) / dx))
                py = int(round(np.dot(vec_from_origin, col_dir) / dy))
                
                # 3. 边界安全检查：防止投影出的点越界（例如在图像外）
                if 0 <= px < closest_ct["cols"] and 0 <= py < closest_ct["rows"]:
                    results[ct_name].add(((px, py), obj_id))

    # 将所有的 set 转换回 list，确保最后可以被 json.dump 正常序列化
    for ct_name in results:
        results[ct_name] = list(results[ct_name])
        
    return results

# 使用示例:
# space_points_dict = {
#     0: [(120.5, -45.2, -938.5), (121.3, -45.2, -938.5)],
#     1: [(120.5, -45.2, -935.5)]
# }
# ct_pixel_mapping = map_physical_to_ct_pixels(space_points_dict, "C:/CT_Folder")

# CT re-sampling, HU threshold, normed.
def preprocess_ct(ct_dicom_dir, output_nii_path, target_spacing=(2.0, 2.0, 2.0), clip_min=-1000, clip_max=1000):
 
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ct_dicom_dir)
    reader.SetFileNames(dicom_names)
    ct_image = reader.Execute()
    
    original_spacing = ct_image.GetSpacing()
    original_size = ct_image.GetSize()
    
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]
    
    print(f"  Original spacing: {original_spacing}, original size: {original_size}")
    print(f"  target spacing: {target_spacing}, new size: {new_size}")
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(ct_image.GetDirection())
    resampler.SetOutputOrigin(ct_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000) # zero if out of VOF
    resampler.SetInterpolator(sitk.sitkBSpline) 
    
    resampled_ct = resampler.Execute(ct_image)
    
    print(f"  HU clip: [{clip_min}, {clip_max}]")
    ct_array = sitk.GetArrayFromImage(resampled_ct)
    
    ct_clipped = np.clip(ct_array, clip_min, clip_max)
    
    print("  Normed to [0, 1]...")
    ct_normalized = (ct_clipped - clip_min) / (clip_max - clip_min)
    
    final_ct_img = sitk.GetImageFromArray(ct_normalized.astype(np.float32))
    final_ct_img.CopyInformation(resampled_ct)
    
    sitk.WriteImage(final_ct_img, output_nii_path)

# def get_ct_pixels_from_seg(seg_path, ct_dir):
#     """
#     直接从 SEG 文件提取前景病灶，并将其映射到对应 CT 图像的像素坐标上。
    
#     参数:
#         seg_path (str): SEG DICOM 文件的完整路径。
#         ct_dir (str): 包含对应 CT DICOM 序列的文件夹路径。
        
#     返回:
#         dict: { "ct_filename.dcm": [(px, py), (px, py), ...] } 
#               其中 px 为列索引 (X)，py 为行索引 (Y)。
#     """
#     # ---------------------------------------------------------
#     # 第一步：扫描 CT 文件夹，建立 Z坐标 -> CT空间参数 的索引字典
#     # ---------------------------------------------------------
#     ct_index = {}
#     for f in os.listdir(ct_dir):
#         if not f.lower().endswith('.dcm'):
#             continue
            
#         ct_file_path = os.path.join(ct_dir, f)
#         # stop_before_pixels=True 仅读取头部，极大提升扫描速度
#         ds_ct = pydicom.dcmread(ct_file_path, stop_before_pixels=True)
        
#         if hasattr(ds_ct, 'ImagePositionPatient'):
#             pos = np.array([float(v) for v in ds_ct.ImagePositionPatient])
#             spacing = [float(v) for v in ds_ct.PixelSpacing] # [dy, dx]
#             orientation = [float(v) for v in ds_ct.ImageOrientationPatient]
            
#             z_val = round(pos[2], 2) # 取两位小数，作为 Z 轴高度的主键
#             ct_index[z_val] = {
#                 "name": f,
#                 "origin": pos,
#                 "dy": spacing[0],
#                 "dx": spacing[1],
#                 "row_dir": np.array(orientation[:3]), # X轴延伸方向
#                 "col_dir": np.array(orientation[3:]), # Y轴延伸方向
#                 "rows": ds_ct.Rows,
#                 "cols": ds_ct.Columns
#             }

#     if not ct_index:
#         raise ValueError(f"在 {ct_dir} 中未找到包含空间信息的 CT 文件！")

#     # ---------------------------------------------------------
#     # 第二步：解析 SEG 文件，提取物理坐标并进行逆向投影
#     # ---------------------------------------------------------
#     ds_seg = pydicom.dcmread(seg_path)
#     seg_array = ds_seg.pixel_array
    
#     # 获取 SEG 的共享空间参数
#     shared_group = ds_seg.SharedFunctionalGroupsSequence[0]
#     seg_orientation = shared_group.PlaneOrientationSequence[0].ImageOrientationPatient
#     seg_row_dir = np.array([float(v) for v in seg_orientation[:3]])
#     seg_col_dir = np.array([float(v) for v in seg_orientation[3:]])
    
#     seg_spacing = shared_group.PixelMeasuresSequence[0].PixelSpacing
#     seg_dy, seg_dx = float(seg_spacing[0]), float(seg_spacing[1])
    
#     results = {}
    
#     # 遍历 SEG 的每一帧
#     for frame_idx in range(seg_array.shape[0]):
#         frame_data = seg_array[frame_idx]
        
#         # 提取前景点 (背景通常为 0)
#         y_indices, x_indices = np.where(frame_data != 0)
#         if len(y_indices) == 0:
#             continue
            
#         # 获取当前帧的绝对物理原点
#         frame_group = ds_seg.PerFrameFunctionalGroupsSequence[frame_idx]
#         seg_origin = np.array([float(v) for v in frame_group.PlanePositionSequence[0].ImagePositionPatient])
#         seg_z = round(seg_origin[2], 2)
        
#         # 寻找 Z 轴高度最匹配的 CT 切片 (容差设为 2.0 mm，可根据 PET 层厚调整)
#         matched_ct_info = None
#         min_diff = float('inf')
#         for ct_z, info in ct_index.items():
#             diff = abs(ct_z - seg_z)
#             if diff < min_diff and diff <= 2.0:
#                 min_diff = diff
#                 matched_ct_info = info
                
#         if not matched_ct_info:
#             # 如果该帧 SEG 在物理高度上找不到对应的 CT，则跳过
#             continue
            
#         ct_name = matched_ct_info["name"]
#         if ct_name not in results:
#             results[ct_name] = set() # 使用 set 自动去重
            
#         ct_origin = matched_ct_info["origin"]
#         ct_dx, ct_dy = matched_ct_info["dx"], matched_ct_info["dy"]
#         ct_row_dir = matched_ct_info["row_dir"]
#         ct_col_dir = matched_ct_info["col_dir"]
#         max_y, max_x = matched_ct_info["rows"], matched_ct_info["cols"]

#         # 针对该帧中的每一个病灶像素进行坐标变换
#         for y, x in zip(y_indices, x_indices):
#             # 1. SEG 像素 -> 绝对物理坐标 (LPS)
#             phys_pt = seg_origin + (x * seg_dx * seg_row_dir) + (y * seg_dy * seg_col_dir)
            
#             # 2. 绝对物理坐标 -> CT 像素坐标 (向量点乘投影)
#             # 计算物理点相对于 CT 原点的向量
#             vector_from_ct_origin = phys_pt - ct_origin
            
#             # 将向量投影到 CT 的行/列方向上，并除以像素间距
#             ct_px = int(round(np.dot(vector_from_ct_origin, ct_row_dir) / ct_dx))
#             ct_py = int(round(np.dot(vector_from_ct_origin, ct_col_dir) / ct_dy))
            
#             # 边界保护：确保算出的像素点没有飞出 CT 图像的范围
#             if 0 <= ct_px < max_x and 0 <= ct_py < max_y:
#                 results[ct_name].add((ct_px, ct_py))

#     # 将 set 转回 list 以便后续支持 JSON 序列化
#     for key in results:
#         results[key] = list(results[key])
        
#     return results

if __name__=='__main__':
    seg_rst = sp.extract_seg_physical_coords("raw_datasets/PSMA_1bbf6768fbbf5228/04-19-2000-NA-PETCT whole-body PSMA-65511/300.000000-Segmentation - No tumor lesions detected-67379/1-1.dcm")
    rst = map_physical_to_ct_pixels(seg_rst, "raw_datasets/PSMA_1bbf6768fbbf5228/04-19-2000-NA-PETCT whole-body PSMA-65511/2.000000-CT-81067")
    
    for key in rst.keys():
        print(f"@@ {key} @@")
        print(rst[key])
        print("----")
    
    # rst = get_ct_pixels_from_seg("raw_datasets/PSMA_2e5119d4ac37d41d/06-29-1999-NA-PETCT whole-body PSMA-90640/300.000000-Segmentation - Tumor lesions detected-49927/1-1.dcm",
    #                        "raw_datasets/PSMA_2e5119d4ac37d41d/06-29-1999-NA-PETCT whole-body PSMA-90640/2.000000-CT-70019")
    
    # for key in rst.keys():
    #     print(f"@@ {key} @@")
    #     print(rst[key])
    #     print("----")
    n = sum(len(lst) for lst in rst.values())
    print(n)
