import json
import os

def safe_read_json(file_path):
    """安全读取 JSON 文件，防 0KB 报错与 BOM 编码问题"""
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 -> {file_path}")
        return {}
        
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()
        
    if not content:
        print(f"⚠️ 警告: 文件完全是空的 (0 KB) -> {file_path}")
        return {}
        
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 格式错误: '{file_path}'。详细报错: {e}")
        return {}

def process_and_map_json_with_coords(info_json_path, map_json_path):
    """
    升级版重组函数：保留所有像素坐标信息
    返回结构形如：
    {
        0: {                        # 映射后的患者 ID (原 "1" -> 0)
            234: [                  # 映射后的切片 ID (原 "1-235.dcm" -> 234)
                [[311, 261], 1],    # 坐标及正负标签
                [[311, 265], 1]
            ],
            233: [ ... ]
        }
    }
    """
    info_data = safe_read_json(info_json_path)
    map_data = safe_read_json(map_json_path)
    
    if not info_data or not map_data:
        print("🚨 数据读取失败，管线终止。")
        return {}

    # 1. 建立反向查找字典: { "study_id": int(key) - 1 }
    study_to_new_key = {}
    for map_key, map_val in map_data.items():
        study_id = map_val.get("study_id")
        if study_id:
            study_to_new_key[study_id] = int(map_key) - 1

    result_dict = {}

    # 2. 遍历重组数据，这次我们要保留 Value
    for project_id, studies in info_data.items():
        for study_id, dcm_dict in studies.items():
            
            if study_id not in study_to_new_key:
                # print(f"⚠️ 警告: study_id '{study_id}' 在 map-json 中找不到映射。")
                continue
                
            new_key = study_to_new_key[study_id]
            
            # 如果这个新患者 ID 还没被创建，就初始化一个空字典
            if new_key not in result_dict:
                result_dict[new_key] = {}
            
            # 提取切片序号，并将原始的坐标列表原封不动地挂载上去
            for dcm_name, coords_list in dcm_dict.items():
                try:
                    # 解析 "1-235.dcm" -> "235" -> 234
                    x_str = dcm_name.split('-')[1].split('.')[0]
                    slice_idx = int(x_str) - 1
                    
                    # 【核心改变】：将整个坐标 list 直接赋给这个 slice_idx
                    result_dict[new_key][slice_idx] = coords_list
                    
                except Exception as e:
                    print(f"⚠️ 警告: 无法解析文件名 '{dcm_name}' -> {e}")

    # (可选) 强迫症福音：对内部的切片序号进行升序排序，方便人类查看 JSON 
    # Python 3.7+ 字典是有序的
    for patient_id in result_dict:
        sorted_slices = dict(sorted(result_dict[patient_id].items()))
        result_dict[patient_id] = sorted_slices

    return result_dict

# --- 运行示例 ---
# final_dict = process_and_map_json_with_coords("info.json", "map.json")
# 
# 如果你想测试看看 0号病人 第234张切片 的第一个正样本点：
# print(final_dict[0][234][0])



if __name__=='__main__':
    my_dict = process_and_map_json_with_coords("samsyn_json_metadata/lesion_ct_pixel_coords.json", "samsyn_json_metadata/rename_mapping.json")
    file_path = "samsyn_json_metadata/nii_idx_with_prompts_coords.json"
    with open(file_path, "w", encoding="utf-8") as f:
        # 2. 把对象 f 传给 json.dump
        json.dump(my_dict, f, indent=4, ensure_ascii=False)
    