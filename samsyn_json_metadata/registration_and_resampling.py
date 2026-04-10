import os
import json
import pydicom
import SimpleITK as sitk

def get_clean_dicom_info(filepath):
    """
    智能提取：读取 Z 坐标、UID、文件名，并拦截非 3D 扫描数据。
    """
    ds = pydicom.dcmread(filepath, stop_before_pixels=True)
    
    # 过滤掉定位图、报告图等干扰项
    img_type = str(ds.get('ImageType', [''])).upper()
    if any(k in img_type for k in ['LOCALIZER', 'SCREEN SAVE', 'DOSE']):
        return None
        
    z = 0.0
    if 'ImagePositionPatient' in ds and len(ds.ImagePositionPatient) >= 3:
        z = float(ds.ImagePositionPatient[2])
    elif 'SliceLocation' in ds:
        z = float(ds.SliceLocation)
        
    uid = str(ds.get('SeriesInstanceUID', 'unknown'))
    filename = os.path.basename(filepath)
    
    return z, uid, filename, filepath

def get_largest_series_metadata(directory):
    """
    找出切片最多的 Series，并返回按 Z 轴排序后的元数据列表。
    返回格式: [(z, uid, filename, path), ...]
    """
    series_dict = {}
    for f in os.listdir(directory):
        if not f.lower().endswith('.dcm'): continue
        path = os.path.join(directory, f)
        try:
            info = get_clean_dicom_info(path)
            if not info: continue
            
            z, uid, filename, filepath = info
            if uid not in series_dict:
                series_dict[uid] = []
            series_dict[uid].append((z, uid, filename, filepath))
        except Exception:
            continue
            
    if not series_dict:
        return []
        
    # 选取切片最多的 UID 组
    best_uid = max(series_dict.keys(), key=lambda k: len(series_dict[k]))
    sorted_metadata = sorted(series_dict[best_uid], key=lambda x: x[0])
    return sorted_metadata

def resample_ct_to_pet(sitk_ct, sitk_pet):
    """配准重采样：CT 向 PET 对齐，并进行类型安全转换"""
    sitk_ct = sitk.Cast(sitk_ct, sitk.sitkFloat32)
    sitk_pet = sitk.Cast(sitk_pet, sitk.sitkFloat32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_pet)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000.0) 
    resampler.SetTransform(sitk.Transform()) 
    
    matched_ct = resampler.Execute(sitk_ct)
    return sitk.Cast(matched_ct, sitk.sitkInt16)

def process_study(study_path):
    """处理 Study 文件夹，成功则返回 (1, 1, 映射字典)，失败则返回 (0, 0, None)"""
    pet_dir = ct_dir = None
    
    for d in os.listdir(study_path):
        full_d = os.path.join(study_path, d)
        if not os.path.isdir(full_d): continue
        name_upper = d.upper()
        # 排除 Segmentation 干扰
        if "SEG" in name_upper: continue
        if "PET" in name_upper: pet_dir = full_d
        elif "CT" in name_upper: ct_dir = full_d
    
    if not pet_dir or not ct_dir:
        return 0, 0, None

    try:
        # 1. 获取排序后的 PET 和 CT 元数据
        pet_meta = get_largest_series_metadata(pet_dir)
        ct_meta = get_largest_series_metadata(ct_dir)

        if not pet_meta or not ct_meta:
            return 0, 0, None

        # 2. 提取路径用于 SimpleITK 读取
        pet_paths = [m[3] for m in pet_meta]
        ct_paths = [m[3] for m in ct_meta]

        # 3. 读取并配准
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(pet_paths)
        sitk_pet = reader.Execute()
        
        reader.SetFileNames(ct_paths)
        sitk_ct = reader.Execute()

        matched_ct = resample_ct_to_pet(sitk_ct, sitk_pet)

        # 4. 保存 NIfTI 文件
        sitk.WriteImage(matched_ct, os.path.join(study_path, "ct.nii.gz"))
        sitk.WriteImage(sitk_pet, os.path.join(study_path, "pet.nii.gz"))

        # 5. 生成映射字典（不再在这里保存为文件，而是作为变量返回）
        mapping_dict = {int(idx): (m[1], m[2]) for idx, m in enumerate(pet_meta)}
        
        print(f"✅ 已完成: {os.path.basename(study_path)} | 帧数: {len(mapping_dict)}")
        return 1, 1, mapping_dict 

    except Exception as e:
        print(f"❌ 失败 [{os.path.basename(study_path)}]: {e}")
        return 0, 0, None

def run_pipeline(root_path):
    """主程序入口"""
    if not os.path.exists(root_path):
        print(f"❌ 路径不存在: {root_path}")
        return
        
    total_ct = total_pet = 0
    # 新增：用于收集所有 Study 映射信息的全局大字典
    global_mapping_info = {} 
    
    print("🚀 启动自动化配准与元数据追踪流水线...")
    
    for proj in os.listdir(root_path):
        proj_p = os.path.join(root_path, proj)
        if not os.path.isdir(proj_p): continue
        
        for study in os.listdir(proj_p):
            study_p = os.path.join(proj_p, study)
            if not os.path.isdir(study_p): continue
            
            # 接收返回的三个值
            c, p, study_mapping = process_study(study_p)
            total_ct += c
            total_pet += p
            
            # 如果成功生成了映射，则将其挂载到全局大字典下（以 Study ID 为键）
            if study_mapping is not None:
                global_mapping_info[study] = study_mapping

    # 所有目录遍历完毕后，将全局字典一次性写入根目录
    output_json_path = os.path.join(root_path, "mapping_info.json")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(global_mapping_info, f, indent=4, ensure_ascii=False)
        print(f"\n📁 成功汇聚所有元数据，全局映射文件已保存至: {output_json_path}")
    except Exception as e:
        print(f"\n❌ 保存全局映射文件失败: {e}")

    print("\n" + "="*40)
    print(f"📊 任务总结:\n   CT NIfTI: {total_ct}\n   PET NIfTI: {total_pet}")
    print("="*40)


# this function is used for check if ct files or PET files exist "huge gap",
# wihch leading the warning: WARNING: In D:\a\SimpleITK\SimpleITK\bld\ITK-prefix\include\ITK-5.4\itkImageSeriesReader.hxx, line 478
# ImageSeriesReader (000001D2D237DDE0): Non uniform sampling or missing slices detected,  maximum nonuniformity:15.1499

def get_z_coordinate(filepath):
    """提取 DICOM 文件的绝对 Z 轴物理坐标"""
    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        if 'ImagePositionPatient' in ds and len(ds.ImagePositionPatient) >= 3:
            return float(ds.ImagePositionPatient[2])
        elif 'SliceLocation' in ds:
            return float(ds.SliceLocation)
    except Exception:
        pass
    return None

def check_adjacent_z_spacing(root_dir):
    """
    遍历目录，以文件夹为单位，计算并打印相邻 dcm 文件的 Z 轴绝对差值。
    """
    if not os.path.exists(root_dir):
        print(f"❌ 目录不存在: {root_dir}")
        return

    # 遍历所有层级的目录
    for dirpath, _, filenames in os.walk(root_dir):
        # 收集当前文件夹下的所有有效 dcm 文件及其 Z 坐标
        dcm_files = []
        for f in filenames:
            if f.lower().endswith('.dcm'):
                filepath = os.path.join(dirpath, f)
                z_val = get_z_coordinate(filepath)
                if z_val is not None:
                    dcm_files.append((z_val, f))
        
        # 如果当前文件夹下包含超过 1 个 dcm 文件，则进行比对
        if len(dcm_files) > 1:
            # 严格按照 Z 轴坐标进行物理排序
            dcm_files.sort(key=lambda x: x[0])
            
            print(f"\n📂 正在分析文件夹: {dirpath}")
            print("-" * 60)
            
            # 遍历第 n 个和 n+1 个文件，计算差值
            for i in range(len(dcm_files) - 1):
                z_current, name_current = dcm_files[i]
                z_next, name_next = dcm_files[i + 1]
                
                # 计算绝对差值
                diff = abs(z_current - z_next)
                
                # 打印结果，保留 4 位小数以便观察细微的精度误差
                print(f"[{i:03d} -> {i+1:03d}] {name_current} 与 {name_next} | "
                      f"Z轴差值: {diff:.4f} mm")
            print("-" * 60)

# --- 运行示例 ---
# my_target_folder = "你的/目标/测试路径"
# check_adjacent_z_spacing(my_target_folder)

if __name__ == "__main__":
    run_pipeline("./PSMA-PET-CT-Lesions")
    #check_adjacent_z_spacing("C:\\Users\\xiaow\\Documents\\Work\\NBIA-PSMA-manifest-1772126181965\\PSMA-PET-CT-Lesions\\PSMA_738bc5d9946240f3\\09-26-1997-NA-PETCT whole-body PSMA-54011\\2.000000-CT-31316")
    #check_adjacent_z_spacing("C:\\Users\\xiaow\\Documents\\Work\\NBIA-PSMA-manifest-1772126181965\\PSMA-PET-CT-Lesions\\PSMA_3959d1c381a5bcd6\\10-05-1998-NA-PETCT whole-body PSMA-28057\\6.000000-PET-72081")
