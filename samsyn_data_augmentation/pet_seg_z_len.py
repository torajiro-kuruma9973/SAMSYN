import os
import json
import SimpleITK as sitk

def compare_nifti_z_lengths(input1_dir, input2_dir, output_json_path):
    """
    遍历两个文件夹中对应的 .nii.gz 文件，极速读取其 Z 轴长度，并导出为 JSON。
    
    参数:
        input1_dir (str): 第一个文件夹路径
        input2_dir (str): 第二个文件夹路径 (要求文件名与 input1 对应)
        output_json_path (str): 输出的 JSON 文件路径
    """
    # 1. 找出 input1 中所有的 NIfTI 文件
    nii_files = [f for f in os.listdir(input1_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    if not nii_files:
        print(f"⚠️ 在 {input1_dir} 中没有找到任何 NIfTI 文件！")
        return

    print(f"🚀 开始极速扫描文件头信息 (共 {len(nii_files)} 个文件)...")
    
    result_dict = {}
    missing_count = 0
    error_count = 0
    
    # 预先实例化一个 Reader，循环复用，速度更快
    reader = sitk.ImageFileReader()
    
    # 2. 遍历比对
    for filename in nii_files:
        path1 = os.path.join(input1_dir, filename)
        path2 = os.path.join(input2_dir, filename)
        
        # 提取前面的数字部分作为 Key (例如 "99.nii.gz" -> "99")
        # 如果后缀是 .nii.gz，去掉后 7 个字符；如果是 .nii，去掉后 4 个字符
        ext_len = 7 if filename.endswith('.nii.gz') else 4
        file_key = filename[:-ext_len]
        
        # 检查 input2 中是否存在对应文件
        if not os.path.exists(path2):
            print(f"   ⚠️ 找不到配对文件跳过: {filename} 在 input2 中不存在！")
            missing_count += 1
            continue
            
        try:
            # --- 🌟 核心提速技巧：仅读取 Header ---
            
            # 读取文件 1 的头部
            reader.SetFileName(path1)
            reader.ReadImageInformation()
            # SimpleITK 的 Size 返回顺序是 (X, Y, Z)，因此 index 2 是 Z 轴
            z_len1 = reader.GetSize()[2] 
            
            # 读取文件 2 的头部
            reader.SetFileName(path2)
            reader.ReadImageInformation()
            z_len2 = reader.GetSize()[2]
            
            # 存入字典
            # 注意：JSON 原生不支持 Python 的 tuple(元组)，所以这里用 list [a, b] 存储
            result_dict[file_key] = [z_len1, z_len2]
            
        except Exception as e:
            print(f"   ❌ 读取文件 {filename} 的头信息时发生错误: {str(e)}")
            error_count += 1

    # 3. 导出 JSON
    print(f"\n⏳ 正在保存结果到 {output_json_path} ...")
    
    # 确保输出路径的文件夹存在
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4)
        
    print(f"🎉 任务完成！")
    print(f"   ➤ 成功配对并提取: {len(result_dict)} 个")
    if missing_count > 0 or error_count > 0:
        print(f"   ➤ 缺失: {missing_count} 个 | 报错: {error_count} 个")

# ===============================
# 运行示例
# ===============================
if __name__ == "__main__":
    folder_a = "./aug_pet_nii_files"
    folder_b = "./aug_seg_nii_files"
    out_json = "./pet_seg_z_length.json"
    
    compare_nifti_z_lengths(folder_a, folder_b, out_json)