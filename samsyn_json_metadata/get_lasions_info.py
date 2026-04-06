import os
import pydicom
import numpy as np
import SimpleITK as sitk
import json

# this is a script for mapping lasions coords on seg file to ct pix space

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

def extract_seg_physical_coords(seg_path):
    """
    遍历 SEG 文件，提取不为 0 的前景点，并严格按照 DICOM Header 的变换矩阵计算其物理坐标。
    
    参数:
        seg_path (str): 3D volume 的 DICOM SEG 文件路径。
        
    返回:
        dict: { frame_idx (int): [(x, y, z), (x, y, z), ...] } 
              其中坐标均为 float 类型，保留 4 位小数。
    """
    # 1. 读取 DICOM SEG 文件
    ds = pydicom.dcmread(seg_path)
    pixel_array = ds.pixel_array  # 形状通常为 (Frames, Rows, Columns)
    
    # 2. 获取【全局共享】的空间变换参数
    # SEG 文件强制使用多帧结构，公共参数存储在 SharedFunctionalGroupsSequence 中
    shared_group = ds.SharedFunctionalGroupsSequence[0]
    
    # 方向余弦向量 (ImageOrientationPatient)
    # 前 3 个值代表行延伸方向 (Row direction, 对应列索引变化 / X轴)
    # 后 3 个值代表列延伸方向 (Column direction, 对应行索引变化 / Y轴)
    orientation = shared_group.PlaneOrientationSequence[0].ImageOrientationPatient
    row_dir = np.array([float(v) for v in orientation[:3]])
    col_dir = np.array([float(v) for v in orientation[3:]])
    
    # 像素间距 (PixelSpacing): [相邻行的中心距(dy), 相邻列的中心距(dx)]
    spacing = shared_group.PixelMeasuresSequence[0].PixelSpacing
    dy = float(spacing[0])
    dx = float(spacing[1])
    
    results = {}
    
    # 3. 遍历每一帧
    for frame_idx in range(pixel_array.shape[0]):
        frame_data = pixel_array[frame_idx]
        
        # 找到所有不为 0 的像素索引 (y对应行/Row, x对应列/Column)
        y_indices, x_indices = np.where(frame_data != 0)
        
        # 如果全为0 (无病灶)，直接跳过该帧
        if len(y_indices) == 0:
            continue
            
        # 4. 获取【当前帧专属】的物理原点
        # 即当前帧左上角第一个像素 (0,0) 的 LPS 绝对三维坐标
        frame_group = ds.PerFrameFunctionalGroupsSequence[frame_idx]
        origin = np.array([float(v) for v in frame_group.PlanePositionSequence[0].ImagePositionPatient])
        
        # 5. 核心：计算实际物理空间坐标 (利用 Numpy 向量化加速)
        # DICOM 坐标公式: P(x,y) = Origin + (x * dx * RowDir) + (y * dy * ColDir)
        
        # 将 x, y 偏移量转为列向量，形状从 (N,) 变为 (N, 1) 以便与形状为 (3,) 的方向向量广播相乘
        x_offsets = (x_indices * dx)[:, np.newaxis]
        y_offsets = (y_indices * dy)[:, np.newaxis]
        
        # 向量化计算当前帧所有前景点的 3D 坐标
        # world_coords 的形状将是 (N, 3)
        world_coords = origin + (x_offsets * row_dir) + (y_offsets * col_dir)
        # 6. 将 Numpy 数组转换为 Python 原生 list(tuple) 并保留精度
        # 这样在后续作为 JSON 导出时不会报错
        frame_points = [
            (round(pt[0], 4), round(pt[1], 4), round(pt[2], 4)) 
            for pt in world_coords.tolist()
        ]
        
        assert (frame_data[y_indices, x_indices] != 0).all() # make sure all the lasions coords are unzero.
        obj_list = frame_data[y_indices, x_indices].tolist()
        temp = list(zip(frame_points, frame_data[y_indices, x_indices])) # this will auto convert the obj_id into np.uint8.
        results[frame_idx] = [(coords, val.item()) for coords, val in temp] # make obj_id change back

    return results

def convert_to_native_types(obj):
    """
    递归遍历整个数据结构，将所有的 numpy 数据类型转换为纯 Python 类型。
    """
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return [convert_to_native_types(i) for i in obj] # JSON 不支持 tuple，统一转 list
    elif isinstance(obj, set):
        return [convert_to_native_types(i) for i in obj] # JSON 不支持 set，统一转 list
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    elif isinstance(obj, np.generic):
        return obj.item() # 将 np.int64, np.float32 等标量安全提取为 Python 的 int/float
    else:
        return obj

def get_lasions_info_from_ct(root_path, output_json_path):
    all_results = {}

    projects = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    print(f"find {len(projects)} project folders，processing...")

    for project_id in projects:
        project_dir = os.path.join(root_path, project_id)
        
        # 初始化 project_id 字典
        if project_id not in all_results:
            all_results[project_id] = {}
            
        for sub_folder in os.listdir(project_dir):
            study_dir = os.path.join(project_dir, sub_folder)
            
            if not os.path.isdir(study_dir) or "PETCT" not in sub_folder.upper():
                continue
                
            study_id = sub_folder
            petct_dir = study_dir
            
            ct_dir = None
            seg_file_path = None
            
            for leaf_folder in os.listdir(petct_dir):
                leaf_path = os.path.join(petct_dir, leaf_folder)
                if not os.path.isdir(leaf_path):
                    continue
                    
                leaf_upper = leaf_folder.upper()
                
                if "SEGMENTATION" in leaf_upper:
                    seg_files = [f for f in os.listdir(leaf_path) if f.lower().endswith('.dcm')]
                    if seg_files:
                        seg_file_path = os.path.join(leaf_path, seg_files[0])
                elif "CT" in leaf_upper:
                    ct_dir = leaf_path

            if not ct_dir or not seg_file_path:
                print(f"skip [{project_id} -> {study_id}]: lacks of CT folder or SEG folder.")
                continue

            try:
                print(f"processing: {project_id} -> {study_id} ...")
                
                # 严格保留原有的函数调用
                space_info_dict = extract_seg_physical_coords(seg_file_path)
                ct_pixel_coords = map_physical_to_ct_pixels(space_info_dict, ct_dir)
                
                all_results[project_id][study_id] = ct_pixel_coords
                
            except Exception as e:
                print(f"failed: [{project_id} -> {study_id}]: {e}")

    # ---------------------------------------------------------
    # 保存 JSON 部分
    # ---------------------------------------------------------
    try:
        out_dir = os.path.dirname(output_json_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 🚨 关键修改：在 dump 之前，将 all_results 彻底清洗为纯原生 Python 类型
        clean_results = convert_to_native_types(all_results)

        # 使用最原生的 json.dump 保存
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=4)
            
        print(f"\nDone! json file is saved in: {output_json_path}")
    except Exception as e:
        print(f"\nFailed!!: {e}")

# 1. Pick the folders only contain PET files with prefix "1-". (done in Windows)
# 2. Abstract the foreground pix coordinates on CT, by uising seg files.
# save as json file.
get_lasions_info_from_ct("PSMA-PET-CT-Lesions", "./lesion_ct_pixel_coords.json")


# pp.preprocess_pet(ct_dicom_dir, pet_dicom_dir, output_nii_path)

# cp.preprocess_ct(ct_dicom_dir, output_nii_path, target_spacing=(2.0, 2.0, 2.0), clip_min=-1000, clip_max=1000)

# # 4. (可选) 处理 CT 自身 (截断 HU、归一化等，如果是跨患者统一尺寸的话)
# target_ct = process_CT(ct_img)

# # 5. 分别保存，大功告成！
# sitk.WriteImage(target_ct, "patient_01_0000.nii.gz")
# sitk.WriteImage(target_pet, "patient_01_0001.nii.gz")
# sitk.WriteImage(target_seg, "patient_01_seg.nii.gz")