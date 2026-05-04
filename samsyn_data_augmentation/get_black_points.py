import os
import json
import numpy as np
import pydicom

def extract_pet_seg_values_from_root(root_dir, output_json_path):
    """
    自动在根目录及其子目录下寻找 PET 和 SEG 文件夹，匹配 3D SEG DICOM 的每一帧与对应的 PET DICOM 文件，
    提取前景点和大于前景最大值的高亮斑点坐标及数值。
    
    参数:
        root_dir (str): 传入的根目录 (例如 'a')
        output_json_path (str): 输出 JSON 字典的文件路径
    """
    
    print(f"⏳ 步骤 0: 正在深度扫描目录 '{root_dir}' 寻找目标文件夹...")
    pet_dir = None
    seg_dir = None
    
    # 🌟 修改点：使用 os.walk 递归向下扫描，彻底解决层级识别问题
    for current_path, dir_names, file_names in os.walk(root_dir):
        for dir_name in dir_names:
            lower_name = dir_name.lower()
            # 拼接出完整的绝对/相对路径，例如 a/pet
            full_path = os.path.join(current_path, dir_name)
            
            if "segmentation" in lower_name:
                seg_dir = full_path
                print(f"   ➤ 锁定 SEG 文件夹: {full_path}")
            elif "pet" in lower_name:
                pet_dir = full_path
                print(f"   ➤ 锁定 PET 文件夹: {full_path}")
            elif "ct" in lower_name and "segmentation" not in lower_name:
                print(f"   ➤ 发现 CT  文件夹: {full_path} (本次跳过处理)")
                
        # 如果需要的两个文件夹都找到了，提前结束扫描以提升效率
        if pet_dir and seg_dir:
            break

    # 安全检查：确保需要的两个文件夹都找到了
    if not pet_dir or not seg_dir:
        raise FileNotFoundError(f"❌ 在 '{root_dir}' 目录下未能同时找到带有 'PET' 和 'Segmentation' 关键字的文件夹，请检查目录结构！")

    print("\n⏳ 步骤 1: 扫描 PET 文件夹，建立 UID 映射字典...")
    # 通过 SOPInstanceUID 来精准匹配
    pet_uid_to_path = {}
    for filename in os.listdir(pet_dir):
        if filename.endswith(".dcm") or filename.endswith(".IMA"):
            file_path = os.path.join(pet_dir, filename)
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            if hasattr(ds, 'SOPInstanceUID'):
                pet_uid_to_path[ds.SOPInstanceUID] = file_path

    print("⏳ 步骤 2: 读取 SEG 掩码文件...")
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith(".dcm") or f.endswith(".IMA")]
    if not seg_files:
        raise FileNotFoundError(f"❌ 在 {seg_dir} 中没有找到 SEG DICOM 文件！")
    
    seg_path = os.path.join(seg_dir, seg_files[0])
    seg_ds = pydicom.dcmread(seg_path)
    seg_pixel_array = seg_ds.pixel_array 
    
    results_dict = {}

    print(f"⏳ 步骤 3: 开始逐帧分析... (共 {seg_pixel_array.shape[0]} 帧)")
    
    # 遍历 SEG 的每一帧功能组序列
    for frame_idx, frame_item in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
        
        try:
            ref_uid = frame_item.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        except AttributeError:
            print(f"⚠️ 第 {frame_idx} 帧缺少标准的 SourceImage 引用，已跳过。")
            continue
            
        pet_path = pet_uid_to_path.get(ref_uid)
        
        if not pet_path:
            continue
            
        pet_filename = os.path.basename(pet_path)
        
        # 读取对应的 PET 完整数据
        pet_ds = pydicom.dcmread(pet_path)
        pet_img = pet_ds.pixel_array.astype(np.float64)
        
        # 还原真实的物理数值
        slope = getattr(pet_ds, 'RescaleSlope', 1.0)
        intercept = getattr(pet_ds, 'RescaleIntercept', 0.0)
        pet_img = pet_img * slope + intercept
        
        # 提取当前帧的 mask
        seg_frame = seg_pixel_array[frame_idx]
        
        # 找到前景点坐标 (mask > 0)
        fg_y, fg_x = np.where(seg_frame > 0)
        
        if len(fg_y) == 0:
            continue
            
        # 提取前景对应位置的 PET 数值
        fg_values = pet_img[fg_y, fg_x]
        max_fg_val = float(np.max(fg_values)) 
        
        # 找到整个 PET 切片中，严格大于 max_fg_val 的点
        large_y, large_x = np.where(pet_img > max_fg_val)
        large_values = pet_img[large_y, large_x]
        
        print(f"\n✅ 匹配成功: {pet_filename} (对应 SEG 帧 {frame_idx})")
        print(f"   ➤ 找到前景点: {len(fg_y)} 个, 最大数值: {max_fg_val:.4f}")
        print(f"   ➤ 严格大于最大前景值的散点: {len(large_y)} 个")
        
        # 转换为 Python 原生类型
        fg_list = [(int(y), int(x), float(v)) for y, x, v in zip(fg_y, fg_x, fg_values)]
        large_list = [(int(y), int(x), float(v)) for y, x, v in zip(large_y, large_x, large_values)]
        
        # 存入字典
        results_dict[pet_filename] = {
            "foreground": fg_list,
            "large_value": large_list
        }

    print(f"\n⏳ 步骤 4: 正在保存结果到 {output_json_path} ...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
        
    print("🎉 任务圆满完成！")

# ===============================
# 运行示例
# ===============================
if __name__ == "__main__":
    # 你传进去的根目录 (例如 'a' 或者 './a')
    my_root_folder = "PSMA-PET-CT-Lesions/PSMA_0a3fdc59c5e700d8"  
    out_json = "./pet_analysis_result.json"  
    
    extract_pet_seg_values_from_root(my_root_folder, out_json)