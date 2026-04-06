import json
from pathlib import Path
import re
import sys
import json
import re
import sys
from pathlib import Path

# 强制将标准输出编码设为 UTF-8，防止部分 Windows 终端配置异常
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def count_ct_dcm_files(root_dir, mapping_json_path, output_json_path):
    root_path = Path(root_dir)
    
    # 1. 读取 JSON 映射文件
    try:
        with open(mapping_json_path, 'r', encoding='utf-8') as f:
            study_mapping = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    result_dict = {}
    
    # 正则表达式：只匹配 "数字-数字.dcm" 的文件，过滤掉 Scout/定位图
    dcm_pattern = re.compile(r'^\d+-\d+\.dcm$', re.IGNORECASE)

    print("Starting process...")
    
    # 2. 遍历每一个需要处理的病人
    for new_name, info in study_mapping.items():
        study_id = info.get("study_id")
        
        if not study_id:
            continue
            
        dcm_count = 0
        
        # 匹配逻辑: Root -> 任意 Project_ID -> 目标 Study_ID
        # 例如: C:\PSMA_0179419e313f7d8c\05-03-2002-NA-PETCT...
        study_dirs = list(root_path.glob(f"*/{study_id}"))
        
        if study_dirs:
            study_dir = study_dirs[0]
            
            # 找到 Study 目录下，名字包含 'ct' 且不包含 'seg' 的文件夹
            # 比如保留 "4.000000-CT-32196"，排除 "300.000000-Segmentation..."
            ct_dirs = []
            for sub_dir in study_dir.iterdir():
                if sub_dir.is_dir():
                    dir_name_lower = sub_dir.name.lower()
                    if 'ct' in dir_name_lower and 'seg' not in dir_name_lower:
                        ct_dirs.append(sub_dir)
            
            # 遍历找出的 CT 文件夹
            for ct_dir in ct_dirs:
                # 为了方便调试，我们用传统的 for 循环，不弄什么花里胡哨的一行代码了
                for file in ct_dir.rglob('*'):
                    # 条件1: 是个文件
                    # 条件2: 文件名符合 a-xxx.dcm
                    # 条件3: 它的父级路径里绝对没有 seg (双保险防穿透)
                    if (file.is_file() 
                        and dcm_pattern.match(file.name) 
                        and 'seg' not in str(file.parent).lower()):
                        
                        dcm_count += 1
                        
            # 打印安全的英文进度，绝不引发 cp1252 报错
            print(f"Processed Study: {study_id} -> Count: {dcm_count}")
            
        else:
            print(f"Warning: Cannot find folder for {study_id}")
            
        # 写入字典，key 是 json 最外层的新名字 (如 "1")
        result_dict[new_name] = dcm_count

    # 3. 将结果输出保存
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)
        print(f"\nSuccess! Results saved to: {output_json_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

def load_json_to_dict(json_path: str) -> dict:
    """
    读取 JSON 文件并将其转换为 Python 字典。

    参数:
    - json_path (str): JSON 文件的相对或绝对路径。

    返回:
    - dict: 包含 JSON 数据的字典。如果文件不存在或格式错误，则返回空字典 {}。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            return data_dict
            
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 -> {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件格式不正确，无法解析为 JSON -> {json_path}")
        return {}
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        return {}

# ==========================================
# 使用示例：
# ==========================================
if __name__ == "__main__":
    # 你的数据根目录
    MY_ROOT_DIR = "./PSMA-PET-CT-Lesions"  
    
    # 输入的 json 文件，内容假设为: {"study_001": "Patient_A", "study_002": "Patient_B"}
    INPUT_JSON = "./rename_mapping.json"  
    
    # 输出的 json 文件，内容将会是: {"Patient_A": 150, "Patient_B": 120}
    OUTPUT_JSON = "./dcm_counts.json" 
    
    count_ct_dcm_files(MY_ROOT_DIR, INPUT_JSON, OUTPUT_JSON)