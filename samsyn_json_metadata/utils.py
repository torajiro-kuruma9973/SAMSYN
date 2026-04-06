import json
from pathlib import Path
import re
import sys
import json
import re
import sys
from pathlib import Path

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