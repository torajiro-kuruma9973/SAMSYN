import os
import json

def rename_and_map_nifti_pairs(dir_A, dir_B, json_path_C):
    """
    遍历文件夹 A (CT) 和 B (PET)，提取共同的 StudyID，
    将成对的文件重命名为相同的 index (n.nii.gz)，并导出 JSON 映射字典。
    
    参数:
        dir_A (str): 存放 *_ct.nii.gz 的文件夹路径。
        dir_B (str): 存放 *_pet.nii.gz 的文件夹路径。
        json_path_C (str): 输出映射字典 JSON 的绝对或相对路径。
    """
    if not os.path.exists(dir_A) or not os.path.exists(dir_B):
        print("❌ 错误：文件夹 A 或 文件夹 B 不存在！")
        return

    mapping_dict = {}
    current_index = 0

    print("🚀 开始进行成对重命名与映射生成...")

    # 1. 获取 A 文件夹中所有的 ct 文件，并按字母排序（保证每次运行结果的一致性）
    files_in_A = [f for f in os.listdir(dir_A) if f.endswith("_ct.nii.gz")]
    files_in_A.sort()

    for file_A in files_in_A:
        # 2. 提取极其干净的 StudyID
        # 比如把 "01-02-2002-NA...40922_ct.nii.gz" 截断，保留前面的部分
        study_id = file_A.replace("_ct.nii.gz", "")
        
        # 3. 构造旧路径和预测的 B 文件夹文件名
        old_path_A = os.path.join(dir_A, file_A)
        
        file_B = f"{study_id}_pet.nii.gz"
        old_path_B = os.path.join(dir_B, file_B)

        # 🛡️ 严格校验：确保 B 文件夹中确实存在对应的 pet 文件
        if not os.path.exists(old_path_B):
            print(f"  ⚠️ 跳过 {study_id}: 在文件夹 B 中找不到对应的 {file_B}")
            continue

        # 4. 生成全新的数字文件名
        new_filename = f"{current_index}.nii.gz"
        new_path_A = os.path.join(dir_A, new_filename)
        new_path_B = os.path.join(dir_B, new_filename)

        try:
            # 5. 执行物理改名 (不发生文件复制，瞬间完成)
            os.rename(old_path_A, new_path_A)
            os.rename(old_path_B, new_path_B)
            
            # 6. 记录到映射字典中
            mapping_dict[new_filename] = study_id
            
            print(f"✅ 改名成功: {study_id} -> {new_filename}")
            current_index += 1
            
        except Exception as e:
            print(f"  ❌ 重命名 {study_id} 时发生错误: {e}")

    # 7. 将映射字典保存为 JSON
    try:
        # 确保输出 JSON 的父级目录存在
        os.makedirs(os.path.dirname(os.path.abspath(json_path_C)), exist_ok=True)
        
        with open(json_path_C, 'w', encoding='utf-8') as f:
            json.dump(mapping_dict, f, indent=4, ensure_ascii=False)
        print("\n" + "="*50)
        print(f"🎉 全部处理完毕！共完成 {current_index} 对文件的重命名。")
        print(f"📁 映射文件已安全保存至: {json_path_C}")
        print("="*50)
    except Exception as e:
        print(f"\n❌ 保存 JSON 映射文件失败: {e}")


def convert_studyid_to_index(json_A_path, json_B_path, json_C_path):
    """
    读取包含病灶坐标的 A，和包含名称映射的 B。
    将 A 中的 StudyID 替换为 B 中对应的纯数字 index，并输出为 C。
    
    参数:
        json_A_path (str): 包含坐标信息的 JSON (键为 StudyID)。
        json_B_path (str): 包含文件映射的 JSON (键为 n.nii.gz，值为 StudyID)。
        json_C_path (str): 输出的最终 JSON 文件路径。
    """
    if not os.path.exists(json_A_path):
        print(f"❌ 找不到文件 A: {json_A_path}")
        return
    if not os.path.exists(json_B_path):
        print(f"❌ 找不到文件 B: {json_B_path}")
        return

    print("🚀 开始构建反向映射字典并转换 ID...")

    # 1. 加载数据
    try:
        with open(json_A_path, 'r', encoding='utf-8') as f:
            data_A = json.load(f)
        with open(json_B_path, 'r', encoding='utf-8') as f:
            data_B = json.load(f)
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        return

    # 2. 构建反向字典：{ "StudyID": "n" }
    # 遍历 B，将键名 "n.nii.gz" 剥离后缀，只保留纯数字字符串 "n"
    study_to_idx = {}
    for filename, study_id in data_B.items():
        # 用 replace 把 ".nii.gz" 替换为空，提取出纯数字索引
        idx_str = filename.replace('.nii.gz', '')
        study_to_idx[study_id] = idx_str

    # 3. 执行替换并生成新数据 C
    data_C = {}
    missing_count = 0

    for study_id, frames_data in data_A.items():
        if study_id in study_to_idx:
            # 找到对应的纯数字 index
            new_idx_key = study_to_idx[study_id]
            # 完美继承原本的所有坐标和类别数据
            data_C[new_idx_key] = frames_data
        else:
            # 防御性逻辑：如果 A 里的某个 Study 在 B 中没找到（可能被之前步骤剔除了）
            missing_count += 1
            print(f"  ⚠️ 警告: A 中的 StudyID [{study_id}] 在 B 映射中未找到，已跳过。")

    # 4. 保存为新的 JSON C
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(json_C_path)), exist_ok=True)
        
        with open(json_C_path, 'w', encoding='utf-8') as f:
            json.dump(data_C, f, indent=4, ensure_ascii=False)
            
        print("\n" + "="*50)
        print(f"🎉 转换圆满结束！共成功映射 {len(data_C)} 个患者的数据。")
        if missing_count > 0:
            print(f"⚠️ 注意：有 {missing_count} 个患者未匹配成功被忽略。")
        print(f"📁 最终版训练标签数据已保存至: {json_C_path}")
        print("="*50)
    except Exception as e:
        print(f"\n❌ 保存 JSON C 失败: {e}")


if __name__ == "__main__":
    
    dir_ct = "ct_nii_files"
    dir_pet = "pet_nii_files"
    output_json = "name_mapping.json"
    rename_and_map_nifti_pairs(dir_ct, dir_pet, output_json)

    json_A = "seg_points_info_with_studyID.json"
    json_B = "name_mapping.json"
    json_C = "seg_points_info_with_idx.json"
    
    convert_studyid_to_index(json_A, json_B, json_C)