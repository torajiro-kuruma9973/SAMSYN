import os
import json

def rename_and_map_nifti(pet_dir, ct_dir, json_path="rename_mapping.json"):
    """
    将 PET 和 CT 文件夹中对应的 nii.gz 文件重命名为 1.nii.gz, 2.nii.gz...
    并将原文件名与新文件名的映射关系记录到 JSON 文件中。
    
    注意：此操作是对原文件进行【直接重命名】，不可逆。
    """
    # 确保文件夹存在
    if not os.path.exists(pet_dir) or not os.path.exists(ct_dir):
        raise ValueError("❌ 错误：PET 或 CT 文件夹路径不存在！")

    # 获取 PET 目录下所有的 .nii.gz 文件并按字母排序，保证运行的稳定性
    pet_files = sorted([f for f in os.listdir(pet_dir) if f.endswith('.nii.gz')])
    
    mapping_dict = {}
    counter = 0
    
    print(f"🚀 开始重命名操作，扫描到 {len(pet_files)} 个 PET 文件...")

    for pet_filename in pet_files:
        # 推断对应的 CT 文件名
        # 假设上一步的命名规则是 studyID_pet.nii.gz 和 studyID_ct.nii.gz
        base_study_id = pet_filename.replace('_pet.nii.gz', '')
        ct_filename = f"{base_study_id}_ct.nii.gz"
        
        pet_old_path = os.path.join(pet_dir, pet_filename)
        ct_old_path = os.path.join(ct_dir, ct_filename)
        
        # 检查配对的 CT 文件是否存在
        if os.path.exists(ct_old_path):
            new_filename = f"{counter}.nii.gz"
            
            pet_new_path = os.path.join(pet_dir, new_filename)
            ct_new_path = os.path.join(ct_dir, new_filename)
            
            # --- 核心：直接重命名 (移动/覆盖自身) ---
            os.rename(pet_old_path, pet_new_path)
            os.rename(ct_old_path, ct_new_path)
            
            # 记录到字典中，以新的序号为 Key，方便查阅
            mapping_dict[str(counter)] = {
                "new_name": new_filename,
                "original_pet": pet_filename,
                "original_ct": ct_filename,
                "study_id": base_study_id
            }
            
            counter += 1
        else:
            print(f"⚠️ 警告：找不到 [{pet_filename}] 对应的 CT 文件 [{ct_filename}]，已跳过。")

    # 将映射字典写入 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, indent=4)
        
    print(f"\n🎉 成功重命名 {counter - 1} 对文件！")
    print(f"📝 映射记录已保存至: {json_path}")


if __name__=='__main__':
    PET_DIR = "samsyn_dataset/labels"
    CT_DIR = "samsyn_dataset/data"
    JSON_PATH = "json_metadata/rename_mapping.json"
    rename_and_map_nifti(PET_DIR, CT_DIR, JSON_PATH)