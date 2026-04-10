import os
import json
import pydicom
import SimpleITK as sitk
import math
import datetime
import numpy as np

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

def extract_suv_factor(dicom_path):
    """
    从 PET DICOM 头文件中提取并计算精确的 SUV 转换系数。
    公式: SUV = (Pixel_Value * PatientWeight_in_g) / Decayed_Dose_in_Bq
    """
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)

        # 1. 检查单位，如果已经是 SUV，则系数为 1.0 (某些飞利浦机器或二次导出的数据)
        units = str(ds.get("Units", "")).upper()
        if "SUV" in units or "G/ML" in units:
            return 1.0

        # 2. 提取患者体重并转换为克 (g)
        weight_kg = float(ds.PatientWeight)
        weight_g = weight_kg * 1000.0

        # 3. 提取注射剂量与半衰期
        radio_seq = ds.RadiopharmaceuticalInformationSequence[0]
        injected_dose_bq = float(radio_seq.RadionuclideTotalDose)
        half_life_sec = float(radio_seq.RadionuclideHalfLife)

        # 4. 提取并计算时间差 (秒)
        def parse_dcm_time(time_str):
            time_str = str(time_str).split('.')[0] # 去除小数部分
            return datetime.datetime.strptime(time_str[:6], "%H%M%S")

        inject_time = parse_dcm_time(radio_seq.RadiopharmaceuticalStartTime)
        # 优先使用 AcquisitionTime，其次 SeriesTime
        scan_time_str = ds.get("AcquisitionTime", ds.get("SeriesTime"))
        scan_time = parse_dcm_time(scan_time_str)

        # 跨越午夜的处理
        if scan_time < inject_time:
            scan_time += datetime.timedelta(days=1)

        decay_time_sec = (scan_time - inject_time).total_seconds()

        # 5. 计算衰变校正 (Decay Correction) 极其关键！
        # START: 图像像素已校正到扫描开始时间，所以我们需要把初始剂量衰减到扫描开始时间
        # ADMIN: 图像像素已校正到注射时间，此时直接使用初始剂量即可，无需计算衰减
        decay_correction = str(ds.get("DecayCorrection", "")).upper()
        
        if decay_correction == "ADMIN":
            decayed_dose = injected_dose_bq
        else: # 默认为 START 或 NONE
            # 衰变公式: N = N0 * e^(-ln(2) * t / T1/2)
            decayed_dose = injected_dose_bq * math.exp(-math.log(2) * (decay_time_sec / half_life_sec))

        # 6. 计算最终的 SUV Factor
        suv_factor = weight_g / decayed_dose
        return suv_factor

    except Exception as e:
        print(f"  ⚠️ 警告: 无法从 DICOM 计算 SUV 系数，可能缺失必要 Tag ({e})。回退为 1.0")
        return 1.0

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

def preprocess_sitk_ct(sitk_ct, clip_min=-200.0, clip_max=300.0):
    img_array = sitk.GetArrayFromImage(sitk_ct).astype(np.float32)
    img_array = np.clip(img_array, clip_min, clip_max)
    img_array = (img_array - clip_min) / (clip_max - clip_min + 1e-8)
    processed_sitk = sitk.GetImageFromArray(img_array)
    processed_sitk.CopyInformation(sitk_ct)
    return processed_sitk

def preprocess_sitk_pet(sitk_pet, suv_factor, clip_min=0.0, clip_max=15.0):
    img_array = sitk.GetArrayFromImage(sitk_pet).astype(np.float32)
    
    # 🌟 应用真实的动态物理系数
    img_array = img_array * suv_factor
    
    img_array = np.clip(img_array, clip_min, clip_max)
    img_array = np.log1p(img_array)
    
    log_min = np.log1p(clip_min)
    log_max = np.log1p(clip_max)
    img_array = (img_array - log_min) / (log_max - log_min + 1e-8)
    img_array = np.clip(img_array, 0.0, 1.0)
    
    processed_sitk = sitk.GetImageFromArray(img_array)
    processed_sitk.CopyInformation(sitk_pet)
    
    inv_meta = {
        "suv_factor": float(suv_factor),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "log_min": float(log_min),
        "log_max": float(log_max)
    }
    return processed_sitk, inv_meta

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
    """
    处理 Study 文件夹：配准 -> SUV 预处理 -> 导出 NIfTI。
    ⚠️ 注意：不再在此处保存 JSON 文件，而是将生成的字典 Return 给上层。
    成功返回: (1, 1, 帧映射字典, PET逆向参数字典)
    失败返回: (0, 0, None, None)
    """
    pet_dir = ct_dir = None
    
    for d in os.listdir(study_path):
        full_d = os.path.join(study_path, d)
        if not os.path.isdir(full_d): continue
        name_upper = d.upper()
        if "SEG" in name_upper: continue
        if "PET" in name_upper: pet_dir = full_d
        elif "CT" in name_upper: ct_dir = full_d
    
    if not pet_dir or not ct_dir:
        return 0, 0, None, None

    try:
        # 1. 获取元数据
        pet_meta = get_largest_series_metadata(pet_dir)
        ct_meta = get_largest_series_metadata(ct_dir)

        if not pet_meta or not ct_meta:
            return 0, 0, None, None

        pet_paths = [m[3] for m in pet_meta]
        ct_paths = [m[3] for m in ct_meta]

        # 2. 动态计算真实 SUV Factor
        real_suv_factor = extract_suv_factor(pet_paths[0])

        # 3. 读取序列并执行空间配准
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(pet_paths)
        sitk_pet = reader.Execute()
        
        reader.SetFileNames(ct_paths)
        sitk_ct = reader.Execute()
        matched_ct = resample_ct_to_pet(sitk_ct, sitk_pet)

        # 4. 预处理流水线
        normed_ct = preprocess_sitk_ct(matched_ct)
        normed_pet, pet_inv_meta = preprocess_sitk_pet(sitk_pet, suv_factor=real_suv_factor)

        # 5. 保存 NIfTI 图像
        sitk.WriteImage(normed_ct, os.path.join(study_path, "ct.nii.gz"))
        sitk.WriteImage(normed_pet, os.path.join(study_path, "pet.nii.gz"))

        # 6. 在内存中生成帧映射字典 (强制 idx 为 str，符合 JSON 标准)
        mapping_dict = {str(idx): [m[1], m[2]] for idx, m in enumerate(pet_meta)}
        
        print(f"✅ 已完成预处理: {os.path.basename(study_path)} | 帧数: {len(mapping_dict)}")
        
        # 🌟 核心修改：将这两个极其重要的字典作为变量抛出，交给主函数去收拢
        return 1, 1, mapping_dict, pet_inv_meta 

    except Exception as e:
        print(f"❌ 失败 [{os.path.basename(study_path)}]: {e}")
        return 0, 0, None, None

def run_pipeline(root_path, mapping_output_path, pet_meta_output_path):
    """
    主程序入口：调度底层处理，并统一汇总生成全局 JSON 文件。
    
    参数:
        root_path (str): 包含所有 Project 和 Study 的根目录。
        mapping_output_path (str): 全局帧映射 JSON 的完整输出路径。
        pet_meta_output_path (str): 全局 PET 逆向参数 JSON 的完整输出路径。
    """
    if not os.path.exists(root_path):
        print(f"❌ 路径不存在: {root_path}")
        return
        
    total_ct = total_pet = 0
    
    # 用于收集所有全局信息的两个大字典
    global_mapping_info = {}   
    global_pet_inv_meta = {}   
    
    print("🚀 启动自动化配准与元数据全局追踪流水线...")
    
    for proj in os.listdir(root_path):
        proj_p = os.path.join(root_path, proj)
        if not os.path.isdir(proj_p): continue
        
        for study in os.listdir(proj_p):
            study_p = os.path.join(proj_p, study)
            if not os.path.isdir(study_p): continue
            
            # 接收返回的四个值 (假设 process_study 已在上下文中定义)
            c, p, study_mapping, study_inv_meta = process_study(study_p)
            total_ct += c
            total_pet += p
            
            # 只有成功处理了，才将字典挂载到全局大树上
            if study_mapping is not None and study_inv_meta is not None:
                global_mapping_info[study] = study_mapping
                global_pet_inv_meta[study] = study_inv_meta

    # ==========================================
    # 所有目录遍历完毕，一次性将全局大字典落盘
    # ==========================================
    try:
        # 安全机制：确保 mapping_output_path 的父目录存在
        os.makedirs(os.path.dirname(os.path.abspath(mapping_output_path)), exist_ok=True)
        with open(mapping_output_path, 'w', encoding='utf-8') as f:
            json.dump(global_mapping_info, f, indent=4, ensure_ascii=False)
        print(f"\n📁 成功！全局 [帧映射字典] 已保存至: {mapping_output_path}")
        
        # 安全机制：确保 pet_meta_output_path 的父目录存在
        os.makedirs(os.path.dirname(os.path.abspath(pet_meta_output_path)), exist_ok=True)
        with open(pet_meta_output_path, 'w', encoding='utf-8') as f:
            json.dump(global_pet_inv_meta, f, indent=4, ensure_ascii=False)
        print(f"📁 成功！全局 [PET逆向参数] 已保存至: {pet_meta_output_path}")
        
    except Exception as e:
        print(f"\n❌ 保存全局 JSON 文件失败: {e}")

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
    run_pipeline("./PSMA-PET-CT-Lesions", "./register_info.json", "./pet_inv_meta.json")
    #check_adjacent_z_spacing("C:\\Users\\xiaow\\Documents\\Work\\NBIA-PSMA-manifest-1772126181965\\PSMA-PET-CT-Lesions\\PSMA_738bc5d9946240f3\\09-26-1997-NA-PETCT whole-body PSMA-54011\\2.000000-CT-31316")
    #check_adjacent_z_spacing("C:\\Users\\xiaow\\Documents\\Work\\NBIA-PSMA-manifest-1772126181965\\PSMA-PET-CT-Lesions\\PSMA_3959d1c381a5bcd6\\10-05-1998-NA-PETCT whole-body PSMA-28057\\6.000000-PET-72081")
