import json
import os

def convert_and_offset_mapping(json1_path, json2_path, output_json_path, n):
    """
    根据给定的 Z轴长度 (json1) 和原始映射关系 (json2)，
    计算逆序后的帧映射关系，并为文件名 (key) 加上偏移量 n，最后输出到新 JSON。
    
    参数:
        json1_path (str): 包含各文件 Z 轴长度的 JSON 路径
        json2_path (str): 包含原始帧映射关系的 JSON 路径
        output_json_path (str): 输出的新 JSON 路径
        n (int): 文件名的偏移量 (如 n=100)
    """
    
    print("⏳ 正在读取输入的 JSON 文件...")
    # 1. 读取输入的两个 JSON
    with open(json1_path, 'r', encoding='utf-8') as f1:
        lengths_dict = json.load(f1)
        
    with open(json2_path, 'r', encoding='utf-8') as f2:
        original_mapping = json.load(f2)
        
    new_mapping_dict = {}
    success_count = 0
    missing_count = 0

    print(f"🚀 开始计算逆序索引与偏移 (偏移量 n={n})...")
    # 2. 遍历 json2 的所有原始关系
    for orig_key, frame_dict in original_mapping.items():
        # 安全检查：必须在 json1 中找到对应的长度才能算逆序
        if orig_key not in lengths_dict:
            print(f"   ⚠️ 警告: 键 '{orig_key}' 在 json1 中找不到长度信息，已跳过。")
            missing_count += 1
            continue
            
        # 提取 PET 和 SEG 的原始 Z轴总长度
        # 根据你给的例子：value 类似 [409, 49]
        pet_total_len = lengths_dict[orig_key][0]
        seg_total_len = lengths_dict[orig_key][1]
        
        # 计算新的大 Key (文件名偏移)
        # 比如 "180" -> 180 + 100 = 280 -> "280"
        new_key = str(int(orig_key) + n)
        
        # 初始化这个新文件对应的逆序映射字典
        reversed_frame_dict = {}
        
        # 遍历原来的每一帧对应关系
        for orig_pet_idx_str, orig_seg_idx in frame_dict.items():
            orig_pet_idx = int(orig_pet_idx_str)
            
            # 🌟 核心逆序公式：基于 0 的索引逆转
            # 例: 总长 100，原 idx 65 -> 逆序 100 - 1 - 65 = 34
            rev_pet_idx = pet_total_len - 1 - orig_pet_idx
            rev_seg_idx = seg_total_len - 1 - orig_seg_idx
            
            # 存入新字典 (注意：JSON 的键必须是字符串)
            reversed_frame_dict[str(rev_pet_idx)] = rev_seg_idx
            
        # 挂载到主字典上
        new_mapping_dict[new_key] = reversed_frame_dict
        success_count += 1

    # 3. 导出生成的 json3
    print(f"\n⏳ 正在保存结果到 {output_json_path} ...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(new_mapping_dict, f_out, indent=4)
        
    print(f"🎉 任务圆满完成！")
    print(f"   ➤ 成功转换: {success_count} 个文件映射")
    if missing_count > 0:
        print(f"   ➤ 缺失跳过: {missing_count} 个")

# ===============================
# 运行示例
# ===============================
if __name__ == "__main__":
    j1 = "./pet_seg_z_length.json"       # input1 路径
    j2 = "./seg_points_info_with_idx.json" # input2 路径
    j3 = "./revers_seg_points_info_with_idx.json" # 生成的 json3 路径
    
    # n=100 表示 180.nii.gz 变成 280.nii.gz
    convert_and_offset_mapping(j1, j2, j3, n=1000)