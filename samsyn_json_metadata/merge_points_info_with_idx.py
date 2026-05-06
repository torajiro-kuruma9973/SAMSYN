import json
import os

def merge_json_strictly(json1_path, json2_path, output_json_path):
    """
    严格合并两个 JSON 文件。如果发现顶层键 (如文件名 ID) 有重复，
    会立刻触发报警并终止程序，防止数据被静默覆盖。
    
    参数:
        json1_path (str): 第一个输入 JSON 的路径
        json2_path (str): 第二个输入 JSON 的路径
        output_json_path (str): 合并后输出的新 JSON 路径
    """
    
    print("⏳ 正在读取两个 JSON 文件...")
    
    with open(json1_path, 'r', encoding='utf-8') as f1:
        dict1 = json.load(f1)
        
    with open(json2_path, 'r', encoding='utf-8') as f2:
        dict2 = json.load(f2)
        
    # 🌟 核心防雷机制：提取两个字典的键，并转化为集合 (Set) 计算交集
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    duplicate_keys = keys1.intersection(keys2)
    
    # 只要交集不为空，说明存在重复键，立刻抛出错误并终止
    if duplicate_keys:
        error_msg = f"❌ 致命错误：检测到重复的键值，合并已紧急中止！\n重复的键为: {list(duplicate_keys)}"
        raise ValueError(error_msg)
        
    print("✅ 碰撞检测通过，未发现重复键。准备合并...")
    
    # 字典解包合并 (由于前面已经排除了重复键，这里解包合并绝对安全)
    merged_dict = {**dict1, **dict2}
    
    # 确保输出路径的目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    
    print(f"⏳ 正在保存合并后的结果到 {output_json_path} ...")
    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(merged_dict, f_out, indent=4)
        
    print(f"🎉 任务圆满完成！")
    print(f"   ➤ JSON 1 包含: {len(dict1)} 个键")
    print(f"   ➤ JSON 2 包含: {len(dict2)} 个键")
    print(f"   ➤ 合并后总计: {len(merged_dict)} 个键")

# ===============================
# 运行示例
# ===============================
if __name__ == "__main__":
    j1 = "samsyn_json_metadata/seq_and_reverse_seg_points_info_with_idx.json"
    j2 = "samsyn_json_metadata/left_righy_seg_points_info_with_idx.json"
    j_out = "./samsyn_json_metadata/aug_seg_points_info_with_idx.json"
    
    merge_json_strictly(j1, j2, j_out)