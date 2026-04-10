import os
import json
import pydicom
import numpy as np

def extract_segmentation_points(root_dir, input_json_path, output_json_path):
    """
    遍历目录，解析 SEG 文件，提取前景坐标并聚合成要求的格式：
    {studyID: {pet_idx: [ [[y1, x1], obj1], [[y2, x2], obj2]... ] }}
    注意：没有病灶的阴性样本也会被保留为 {}，这对深度学习极其重要。
    """
    if not os.path.exists(input_json_path):
        print(f"❌ 找不到输入 JSON 文件: {input_json_path}")
        return

    # 1. 加载映射字典
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            mapping_info = json.load(f)
    except Exception as e:
        print(f"❌ 读取 mapping_info.json 失败: {e}")
        return

    output_dict = {}
    print("🚀 启动 SEG 掩码坐标级(原子化)提取引擎...")

    # 2. 遍历项目与 Study
    for proj in os.listdir(root_dir):
        proj_path = os.path.join(root_dir, proj)
        if not os.path.isdir(proj_path): continue

        for study in os.listdir(proj_path):
            study_path = os.path.join(proj_path, study)
            if not os.path.isdir(study_path): continue

            # 如果这个 Study 不在我们的记录里，跳过
            if study not in mapping_info: continue

            pet_dir = seg_dir = None
            for d in os.listdir(study_path):
                full_d = os.path.join(study_path, d)
                if not os.path.isdir(full_d): continue
                name_upper = d.upper()
                
                if "PET" in name_upper: pet_dir = full_d
                elif "SEGMENTATION" in name_upper or "SEG" in name_upper: seg_dir = full_d

            if not pet_dir or not seg_dir:
                continue

            print(f"🔍 正在解析: {study[:30]}...")

            # ==========================================
            # 步骤 A：建立 SOPInstanceUID -> idx 的核心桥梁
            # ==========================================
            idx_mapping = mapping_info[study]  
            sop_to_idx = {}
            study_result = {}  # 存放当前 Study 的结果

            for idx_str, info in idx_mapping.items():
                pet_filename = info[1]
                pet_path = os.path.join(pet_dir, pet_filename)
                
                if os.path.exists(pet_path):
                    try:
                        ds_pet = pydicom.dcmread(pet_path, stop_before_pixels=True)
                        sop_uid = str(ds_pet.SOPInstanceUID)
                        sop_to_idx[sop_uid] = idx_str
                    except Exception:
                        pass

            if not sop_to_idx:
                print(f"  ⚠️ 跳过 (无法提取 PET 的 SOP UID)")
                continue

            # ==========================================
            # 步骤 B：解析 Segmentation 文件并原子化聚合坐标
            # ==========================================
            for seg_file in os.listdir(seg_dir):
                if not seg_file.lower().endswith('.dcm'): continue
                seg_path = os.path.join(seg_dir, seg_file)

                try:
                    ds_seg = pydicom.dcmread(seg_path)
                    pixel_array = ds_seg.pixel_array

                    # 将 2D 单帧也统一扩充为 3D 逻辑处理 (1, H, W)
                    if pixel_array.ndim == 2:
                        pixel_array = pixel_array[np.newaxis, ...]

                    num_frames = pixel_array.shape[0]
                    for f_idx in range(num_frames):
                        ref_sop = None
                        
                        # 寻找该帧引用的原图 UID
                        try:
                            if 'PerFrameFunctionalGroupsSequence' in ds_seg:
                                pffgs = ds_seg.PerFrameFunctionalGroupsSequence[f_idx]
                                ref_sop = str(pffgs.DerivationImageSequence[0]\
                                                   .SourceImageSequence[0]\
                                                   .ReferencedSOPInstanceUID)
                            elif 'ReferencedImageSequence' in ds_seg:
                                ref_sop = str(ds_seg.ReferencedImageSequence[0].ReferencedSOPInstanceUID)
                        except Exception:
                            pass

                        # 如果找到了引用，且能在字典里对上号
                        if ref_sop and ref_sop in sop_to_idx:
                            target_idx = sop_to_idx[ref_sop]
                            frame_pixels = pixel_array[f_idx]

                            # 提取非零像素坐标
                            ys, xs = np.nonzero(frame_pixels)
                            
                            # 如果该帧有前景点（病灶）
                            if len(ys) > 0:
                                # 🌟 核心提取区：为每一个点独立打包 obj
                                points_with_obj = [
                                    [[int(y), int(x)], int(frame_pixels[y, x])] 
                                    for y, x in zip(ys, xs)
                                ]
                                
                                if target_idx not in study_result:
                                    # 初始化列表
                                    study_result[target_idx] = points_with_obj
                                else:
                                    # 如果该 PET 帧已经有记录，直接追加到列表末尾
                                    study_result[target_idx].extend(points_with_obj)

                except Exception as e:
                    print(f"  ❌ 读取 SEG 文件失败 [{seg_file}]: {e}")

            # 🌟 核心修复区：无论有没有病灶，都强制挂载到总字典上！
            # 如果是阴性样本，写入 JSON 的就是一个干净的 `{}`
            output_dict[study] = study_result

    # 3. 标准化保存为 JSON
    try:
        # 保持 indent=4，以确保极佳的可读性
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, indent=4, ensure_ascii=False)
        print(f"\n🎉 完美收官！所有数据（包含珍贵的阴性样本）已精炼至: {output_json_path}")
    except Exception as e:
        print(f"\n❌ 保存 JSON 失败: {e}")


# --- 运行示例 ---
if __name__ == "__main__":
    my_root_dir = "PSMA-PET-CT-Lesions"
    my_input_json = "pet_ct_mapping_info.json"
    my_output_json = "seg_points_info_with_studyID.json"
    extract_segmentation_points(my_root_dir, my_input_json, my_output_json)