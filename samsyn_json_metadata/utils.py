import os
import shutil

def clean_nifti_files(root_dir, dry_run=True):
    """
    递归遍历目录并删除所有的 .nii.gz (和 .nii) 文件。
    
    参数:
        root_dir (str): 目标根目录。
        dry_run (bool): 默认开启安全演习模式。只打印将要删除的文件路径，不会真删。
                        确认无误后，将其设为 False 才会执行毁灭打击。
    """
    if not os.path.exists(root_dir):
        print(f"❌ 目录不存在: {root_dir}")
        return

    deleted_count = 0
    error_count = 0
    
    print(f"🔍 开始深度扫描目录: {root_dir}")
    if dry_run:
        print("🛡️ [安全保护开启] 当前为 Dry-run (演习) 模式，以下文件仅作展示，不会被真正删除：\n")
    else:
        print("⚠️ [警告] 真实删除模式已开启，正在清理...\n")

    # 1. 递归遍历所有目录和子文件夹
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 同时匹配 .nii.gz 和解压后的 .nii
            if filename.lower().endswith('.nii.gz') or filename.lower().endswith('.nii') or filename == "pet_mapping.json":
                filepath = os.path.join(dirpath, filename)
                
                if dry_run:
                    print(f"  🔎 发现目标: {filepath}")
                    deleted_count += 1
                else:
                    try:
                        # 2. 执行真正的文件系统删除
                        os.remove(filepath)
                        print(f"  🗑️ 已粉碎: {filepath}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ❌ 删除失败 [{filepath}]: {e}")
                        error_count += 1

    # 3. 统计报告
    print("\n" + "="*50)
    if dry_run:
        print(f"📊 演习报告: 共扫描到 {deleted_count} 个 NIfTI 文件。")
        print("💡 如果以上路径确认无误（没有误伤原数据），请修改参数运行: ")
        print(f"   clean_nifti_files(root_dir, dry_run=False)")
    else:
        print(f"🏁 清除完毕！成功删除 {deleted_count} 个文件，失败 {error_count} 个。")
    print("="*50)

def clean_json_files(root_dir, dry_run=True):
    """
    递归遍历目录并删除所有的 .nii.gz (和 .nii) 文件。
    
    参数:
        root_dir (str): 目标根目录。
        dry_run (bool): 默认开启安全演习模式。只打印将要删除的文件路径，不会真删。
                        确认无误后，将其设为 False 才会执行毁灭打击。
    """
    if not os.path.exists(root_dir):
        print(f"❌ 目录不存在: {root_dir}")
        return

    deleted_count = 0
    error_count = 0
    
    print(f"🔍 开始深度扫描目录: {root_dir}")
    if dry_run:
        print("🛡️ [安全保护开启] 当前为 Dry-run (演习) 模式，以下文件仅作展示，不会被真正删除：\n")
    else:
        print("⚠️ [警告] 真实删除模式已开启，正在清理...\n")

    # 1. 递归遍历所有目录和子文件夹
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 同时匹配 .nii.gz 和解压后的 .nii
            if filename.lower().endswith('.json') or filename.lower().endswith('pet_inv_meta'):
                filepath = os.path.join(dirpath, filename)
                
                if dry_run:
                    print(f"  🔎 发现目标: {filepath}")
                    deleted_count += 1
                else:
                    try:
                        # 2. 执行真正的文件系统删除
                        os.remove(filepath)
                        print(f"  🗑️ 已粉碎: {filepath}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ❌ 删除失败 [{filepath}]: {e}")
                        error_count += 1

    # 3. 统计报告
    print("\n" + "="*50)
    if dry_run:
        print(f"📊 演习报告: 共扫描到 {deleted_count} 个 json 文件。")
        print("💡 如果以上路径确认无误（没有误伤原数据），请修改参数运行: ")
        print(f"   clean_json_files(root_dir, dry_run=False)")
    else:
        print(f"🏁 清除完毕！成功删除 {deleted_count} 个文件，失败 {error_count} 个。")
    print("="*50)

# rename nii files and move them to specified folders.
def extract_and_rename_nifti(root_dir, ct_output_dir, pet_output_dir, copy_mode=True):
    """
    遍历根目录，寻找各个 Study ID 下的 ct.nii.gz 和 pet.nii.gz，
    将其重命名为 {StudyID}_{modality}.nii.gz 并转移到指定的输出目录。
    
    参数:
        root_dir (str): 原始数据的根目录。
        ct_output_dir (str): CT 文件的集中存放目录。
        pet_output_dir (str): PET 文件的集中存放目录。
        copy_mode (bool): True 为复制模式（保留原文件），False 为剪切移动模式。
    """
    if not os.path.exists(root_dir):
        print(f"❌ 根目录不存在: {root_dir}")
        return

    # 1. 确保输出文件夹存在，如果没有则自动创建
    os.makedirs(ct_output_dir, exist_ok=True)
    os.makedirs(pet_output_dir, exist_ok=True)

    ct_count = 0
    pet_count = 0

    print("🚀 开始扫描并整理 NIfTI 文件...")
    action_name = "复制" if copy_mode else "移动"
    action_func = shutil.copy2 if copy_mode else shutil.move

    # 2. 遍历第一层：Project ID 目录
    for proj_folder in os.listdir(root_dir):
        proj_path = os.path.join(root_dir, proj_folder)
        if not os.path.isdir(proj_path): continue
        
        # 3. 遍历第二层：Study ID 目录
        for study_id in os.listdir(proj_path):
            study_path = os.path.join(proj_path, study_id)
            if not os.path.isdir(study_path): continue
            
            # 定位目标文件
            ct_src = os.path.join(study_path, "ct.nii.gz")
            pet_src = os.path.join(study_path, "pet.nii.gz")
            
            # 4. 处理 CT 文件
            if os.path.exists(ct_src):
                new_ct_name = f"{study_id}_ct.nii.gz"
                ct_dst = os.path.join(ct_output_dir, new_ct_name)
                try:
                    action_func(ct_src, ct_dst)
                    ct_count += 1
                except Exception as e:
                    print(f"  ❌ 处理 CT 失败 [{study_id}]: {e}")

            # 5. 处理 PET 文件
            if os.path.exists(pet_src):
                new_pet_name = f"{study_id}_pet.nii.gz"
                pet_dst = os.path.join(pet_output_dir, new_pet_name)
                try:
                    action_func(pet_src, pet_dst)
                    pet_count += 1
                except Exception as e:
                    print(f"  ❌ 处理 PET 失败 [{study_id}]: {e}")

    # 6. 打印总结报告
    print("\n" + "="*50)
    print(f"🏁 提取归档圆满结束！(模式: {action_name})")
    print(f"   -> 已处理 CT 文件:  {ct_count} 个 存放至 {ct_output_dir}")
    print(f"   -> 已处理 PET 文件: {pet_count} 个 存放至 {pet_output_dir}")
    print("="*50)

import json
import os

def read_json_to_dict(json_path):
    """
    读取 JSON 文件并将其内容存入 Python 字典。
    
    参数:
        json_path (str): JSON 文件的绝对或相对路径。
        
    返回:
        dict: 包含 JSON 数据的字典。如果读取失败，则返回空字典 {}。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"❌ 错误：找不到文件 '{json_path}'")
        return {}

    # 2. 尝试读取并解析 JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
            
    except json.JSONDecodeError as e:
        print(f"❌ 错误：JSON 文件格式不正确 ({json_path})\n详细信息: {e}")
        return {}
    except Exception as e:
        print(f"❌ 读取文件时发生未知错误: {e}")
        return {}


# --- 运行示例 ---
if __name__ == "__main__":
    my_dataset_root = "PSMA-PET-CT-Lesions"
    # clean_json_files(my_dataset_root)
    # clean_json_files(my_dataset_root, dry_run=False)
    # clean_nifti_files(my_dataset_root)
    # clean_nifti_files(my_dataset_root, dry_run=False)

    my_root = "PSMA-PET-CT-Lesions"
    my_ct_out = "ct_nii_files"
    my_pet_out = "pet_nii_files"
    
    # 默认使用安全复制模式，防止原文件丢失
    #extract_and_rename_nifti(my_root, my_ct_out, my_pet_out, copy_mode=True)
    
    # 如果你的硬盘空间吃紧，想直接把文件剪切出来，改成 False 即可：
    extract_and_rename_nifti(my_root, my_ct_out, my_pet_out, copy_mode=False)

    # my_dict = read_json_to_dict("seg_points_info_with_studyID.json")
    # print(my_dict.keys())
