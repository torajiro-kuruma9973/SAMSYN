import json
import samsyn_json_metadata.utils
import samsyn_cfg
import os

def generate_bounded_offset_json(input_dict, offset, max_slices_json_path, output_json_path):
    """
    根据切片数量限制文件，生成带有安全边界的 offset 区间对。
    如果加上 offset 后越界，则用该病人的总切片数代替。
    """
    # 1. 读取含有最大切片数限制的 JSON 文件
    try:
        with open(max_slices_json_path, 'r', encoding='utf-8') as f:
            max_slices_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取限制文件失败: {e}")
        return {}

    result_dict = {}
    
    # 2. 遍历输入的病人切片数据
    for patient_id, slices_dict in input_dict.items():
        patient_id_int = int(patient_id)
        
        # 将 0-based 的 ID 转换为 1-based 的字符串键 (例如 0 -> "1")
        limit_key = str(patient_id_int + 1)
        
        if limit_key not in max_slices_data:
            print(f"⚠️ 警告: 病人 ID {patient_id_int} (对应限制键 '{limit_key}') 未找到，跳过。")
            continue
            
        # 提取该病人的【总层数】
        total_slices = int(max_slices_data[limit_key])
        
        # 因为索引从 0 开始，所以最大的合法切片索引是 总层数 - 1
        max_allowed_index = total_slices - 1

        slice_pairs = []
        for slice_idx in slices_dict.keys():
            start_idx = int(slice_idx)
            target_idx = start_idx + offset
            
            # 🛡️ 核心修改：判断是否越界
            # 如果目标索引超过了最大合法索引，就用总层数代替
            if target_idx > max_allowed_index:
                end_idx = total_slices
            else:
                end_idx = target_idx
            
            slice_pairs.append([start_idx, end_idx])
        
        # 按起始切片从小到大排序
        slice_pairs.sort(key=lambda x: x[0])
        
        # 存入结果字典
        result_dict[patient_id_int] = slice_pairs

    # 3. 保存为 JSON
    try:
        out_dir = os.path.dirname(output_json_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4)
        print(f"✅ 成功！带有边界约束的 offset 字典已保存至: {output_json_path}")
    except Exception as e:
        print(f"❌ 保存 JSON 失败: {e}")
        
    return result_dict

# use this command to run this script: python3 -m samsyn_json_metadata.generate_intervals
if __name__ == "__main__":
    input_dict = samsyn_json_metadata.utils.load_json_to_dict("samsyn_json_metadata/nii_idx_with_prompts_coords.json")
    generate_bounded_offset_json(input_dict, samsyn_cfg.interval_thickness, 
                                "samsyn_json_metadata/dcm_counts.json", 
                                "samsyn_json_metadata/nii_idx_intervals.json")