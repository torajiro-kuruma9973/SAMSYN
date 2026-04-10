import os
import pydicom

def map_seg_frames_to_pet(seg_dcm_path, pet_folder_path):
    """
    读取 3D SEG 文件的每一帧，找出其对应的源图像 SOPInstanceUID，
    并在 PET 文件夹中寻找对应的 dcm 文件名。
    
    返回字典结构: {seg_frame_idx (int, base 0): pet_file_name (str)}
    """
    # 1. 扫描 PET 文件夹，建立【UID -> 文件名】的极速查找表
    print("正在扫描 PET 文件夹建立索引...")
    pet_uid_to_filename = {}
    
    if not os.path.exists(pet_folder_path):
        print(f"❌ PET 文件夹不存在: {pet_folder_path}")
        return {}

    for filename in os.listdir(pet_folder_path):
        if not filename.lower().endswith('.dcm'):
            continue
            
        filepath = os.path.join(pet_folder_path, filename)
        try:
            # 同样使用 stop_before_pixels=True，瞬间读完几百个文件的 Header
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
            uid = str(ds.SOPInstanceUID)
            pet_uid_to_filename[uid] = filename
        except Exception as e:
            print(f"⚠️ 无法读取 PET 文件 [{filename}]: {e}")

    # 2. 读取 3D SEG 文件
    print(f"正在解析 SEG 文件: {os.path.basename(seg_dcm_path)}...")
    try:
        seg_ds = pydicom.dcmread(seg_dcm_path, stop_before_pixels=True)
    except Exception as e:
        print(f"❌ 无法读取 SEG 文件: {e}")
        return {}

    result_dict = {}

    # 3. 提取 SEG 文件中每一帧对应的源图像 UID
    # 在标准的 DICOM SEG 中，每帧的信息存放在 PerFrameFunctionalGroupsSequence 中
    if 'PerFrameFunctionalGroupsSequence' not in seg_ds:
        print("❌ 这不是一个标准的 3D SEG 文件，缺少 PerFrameFunctionalGroupsSequence。")
        return {}

    # enumerate 默认就是 base 0 (从 0 开始计数)
    for frame_idx, frame_group in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
        try:
            # DICOM 标准路径：从每一帧的宏属性中逐级找到 ReferencedSOPInstanceUID
            ref_uid = str(frame_group
                          .DerivationImageSequence[0]
                          .SourceImageSequence[0]
                          .ReferencedSOPInstanceUID)
            
            # 去我们第一步建立的“字典”里查名字
            if ref_uid in pet_uid_to_filename:
                result_dict[frame_idx] = pet_uid_to_filename[ref_uid]
            else:
                print(f"⚠️ 警告: SEG 的第 {frame_idx} 帧指向了 UID [{ref_uid}]，但在 PET 文件夹中未找到匹配的文件！")
                
        except AttributeError:
            # 有些非标的数据集结构可能略有不同
            print(f"⚠️ 警告: 无法在 SEG 的第 {frame_idx} 帧中提取到源图像 UID。")

    print(f"✅ 匹配完成！成功映射 {len(result_dict)} 帧。")
    return result_dict

if __name__ == "__main__":
    my_seg_path = "D:\\D_Work\\Datasets\\NBIA-PSMA-manifest-1772126181965\\PSMA-PET-CT-Lesions\\PSMA_01a52e26ce5b5e26\\10-15-1999-NA-PETCT whole-body PSMA-89953\\300.000000-Segmentation - Tumor lesions detected-87231\\1-1.dcm"
    my_pet_dir = "D:\\D_Work\\Datasets\\NBIA-PSMA-manifest-1772126181965\\PSMA-PET-CT-Lesions\\PSMA_01a52e26ce5b5e26\\10-15-1999-NA-PETCT whole-body PSMA-89953\\102.000000-PET-72841"
    
    matched_dict = map_seg_frames_to_pet(my_seg_path, my_pet_dir)
    
    print(matched_dict)
    