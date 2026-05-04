import os
import numpy as np
import SimpleITK as sitk

def reverse_nifti_z_axis(input_dir, output_dir):
    """
    遍历文件夹，读取所有 .nii.gz 文件，将其 Z 轴（切片顺序）逆序，并保存到新路径。
    
    参数:
        input_dir (str): 存放原始 nii.gz 文件的输入文件夹
        output_dir (str): 存放逆序后 nii.gz 文件的输出文件夹
    """
    # 1. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 找出所有的 NIfTI 文件
    nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    if not nii_files:
        print(f"⚠️ 在 {input_dir} 中没有找到任何 NIfTI 文件！")
        return

    print(f"🚀 开始处理，共找到 {len(nii_files)} 个文件...\n")
    
    success_count = 0
    
    # 3. 逐个处理
    for filename in nii_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # 读取图像
            sitk_img = sitk.ReadImage(input_path)
            
            # SimpleITK 提取出的 NumPy 数组形状是 (Z, Y, X)
            img_array = sitk.GetArrayFromImage(sitk_img)
            
            # 🌟 核心操作：在 Z 轴 (axis=0) 上进行逆序切片
            # 注意：加上 .copy() 是为了让数组在内存中重新连续分布，
            # 否则 SimpleITK 在转回 Image 时可能会因为内存步长异常而报错或产生雪花点。
            reversed_array = img_array[::-1, :, :].copy()
            
            # 将 NumPy 数组转回 SimpleITK Image
            reversed_sitk_img = sitk.GetImageFromArray(reversed_array)
            
            # 🌟 极其关键：复制原始图像的空间元数据（坐标原点、像素间距、方向矩阵）
            reversed_sitk_img.CopyInformation(sitk_img)
            
            # 保存到输出目录
            sitk.WriteImage(reversed_sitk_img, output_path)
            
            success_count += 1
            print(f"✅ 成功翻转并保存: {filename}")
            
        except Exception as e:
            print(f"❌ 处理 {filename} 时发生错误: {str(e)}")
            
    print(f"\n🎉 任务完成！共成功处理 {success_count} / {len(nii_files)} 个文件。")
    print(f"📂 输出路径: {os.path.abspath(output_dir)}")

def offset_nifti_filenames(input_dir, n):
    """
    原地批量修改目录下的纯数字 .nii.gz 文件名，为原数字加上一个偏移量 n。
    采用两步临时重命名法，绝对防止文件名覆盖冲突。
    
    参数:
        input_dir (str): 包含 .nii.gz 文件的文件夹路径
        n (int): 要加上的偏移量 (可以为正数也可以为负数)
    """
    # 1. 找出目录下所有以 .nii.gz 结尾的文件
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    
    if not all_files:
        print(f"⚠️ 在 {input_dir} 中没有找到任何 .nii.gz 文件。")
        return
        
    rename_tasks = []
    
    print(f"🚀 开始扫描并计算新文件名...")
    # 2. 预先计算好所有的 [旧名字 -> 临时名字 -> 新名字]
    for filename in all_files:
        # 去掉结尾的 7 个字符 ".nii.gz"，提取前面的字符串
        base_name = filename[:-7]
        
        try:
            # 尝试将其转换为整数
            old_num = int(base_name)
            new_num = old_num + n
            new_filename = f"{new_num}.nii.gz"
            
            # 构建完整路径
            old_path = os.path.join(input_dir, filename)
            tmp_path = os.path.join(input_dir, f"{new_filename}.tmp") # 临时防冲突后缀
            final_path = os.path.join(input_dir, new_filename)
            
            rename_tasks.append((old_path, tmp_path, final_path, old_num, new_num))
            
        except ValueError:
            # 如果文件名不是纯数字 (比如 "patient.nii.gz")，就跳过不处理
            print(f"   ⏭️ 忽略非纯数字命名的文件: {filename}")

    if not rename_tasks:
        print("⚠️ 没有找到纯数字命名的 .nii.gz 文件需要修改。")
        return

    print(f"🛡️ 开启防碰撞重命名 (共 {len(rename_tasks)} 个文件)...")
    
    # 3. 第一步：把所有目标文件都重命名为带有 .tmp 的临时文件
    # 这样可以彻底清空原来的名字空间，防止后面改名时发生碰撞
    for old_path, tmp_path, _, _, _ in rename_tasks:
        os.rename(old_path, tmp_path)
        
    # 4. 第二步：去掉 .tmp 后缀，落位成最终的新名字
    for _, tmp_path, final_path, old_num, new_num in rename_tasks:
        os.rename(tmp_path, final_path)
        print(f"   ➤ 重命名: {old_num}.nii.gz  -->  {new_num}.nii.gz")
        
    print(f"\n🎉 任务圆满完成！成功将 {len(rename_tasks)} 个文件偏移了 {n}。")

# ===============================
# 运行示例
# ===============================
if __name__ == "__main__":
    # 替换为你实际的文件夹路径
    source_folder = "./pet_nii_files"
    target_folder = "./aug_pet_nii_files"
    my_folder = "./aug_ct_nii_files"
    #reverse_nifti_z_axis(source_folder, target_folder)
    offset_nifti_filenames(my_folder, n=1000)